"""
PushT Multitask DataModule
Loads data from zarr data file with keypoint and n_contacts observations.

The multitask variant includes:
  - img:        (T, 96, 96, 3) uint8 RGB images
  - state:      (T, 5) float32  [agent_pos(2), block_pose(3)]
  - action:     (T, 2) float32  2D movement commands
  - keypoint:   (T, 9, 2) float32  9 keypoints × 2D coords of the T-block
  - n_contacts: (T, 1) float32  number of contacts
"""

from typing import Dict, Optional
import copy

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.nn.data import ReplayBuffer
from src.nn.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from src.nn.common.normalize_util import get_image_range_normalizer
from src.nn.common.pytorch_util import dict_apply
from src.nn.modules import LinearNormalizer
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PushtMultitaskDataset(Dataset):
    """PushT multitask dataset with keypoint and n_contacts observations."""

    def __init__(
        self,
        zarr_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        test_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["img", "state", "action", "keypoint", "n_contacts"]
        )

        val_mask, test_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        train_mask = ~(val_mask | test_mask)
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.mask = train_mask
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.mask = self.val_mask
        return val_set

    def get_test_dataset(self):
        test_set = copy.copy(self)
        test_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.test_mask,
        )
        test_set.mask = self.test_mask
        return test_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :2],
            # keypoint: (N, 9, 2) — flatten last 2 dims for normalizer fitting
            "keypoint": self.replay_buffer["keypoint"].reshape(-1, 18),
            "n_contacts": self.replay_buffer["n_contacts"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][:, :2].astype(np.float32)          # (T, 2)
        image = np.moveaxis(sample["img"], -1, 1) / 255                 # (T, 3, 96, 96)
        keypoint = sample["keypoint"].reshape(sample["keypoint"].shape[0], -1).astype(np.float32)  # (T, 18)
        n_contacts = sample["n_contacts"].astype(np.float32)            # (T, 1)

        return {
            "obs": {
                "image": image,
                "agent_pos": agent_pos,
                "keypoint": keypoint,
                "n_contacts": n_contacts,
            },
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)


class PushtMultitaskDataModule(LightningDataModule):
    """Lightning DataModule for PushT multitask dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        pad_before: int = 0,
        pad_after: int = 0,
        horizon: int = 1,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.horizon = horizon
        self.max_train_episodes = max_train_episodes

        self.train_dataset = PushtMultitaskDataset(
            zarr_path=self.data_dir,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            seed=self.seed,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            max_train_episodes=self.max_train_episodes,
        )
        log.info(f"PushtMultitaskDataModule: Loaded dataset from {data_dir}")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.val_dataset = self.train_dataset.get_validation_dataset()
        if stage == "test" or stage is None:
            self.test_dataset = self.train_dataset.get_test_dataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )


if __name__ == "__main__":
    print("Testing PushtMultitaskDataModule...")
    print("=" * 60)

    dm = PushtMultitaskDataModule(
        data_dir="data/pusht_multitask",
        batch_size=4,
        horizon=16,
        num_workers=0,
    )
    dm.setup("fit")

    train_episodes = set(np.where(dm.train_dataset.train_mask)[0])
    val_episodes = set(np.where(dm.train_dataset.val_mask)[0])
    test_episodes = set(np.where(dm.train_dataset.test_mask)[0])

    print(f"  Train: {len(train_episodes)} episodes")
    print(f"  Val:   {len(val_episodes)} episodes")
    print(f"  Test:  {len(test_episodes)} episodes")
    assert len(train_episodes & val_episodes) == 0
    assert len(train_episodes & test_episodes) == 0
    assert len(val_episodes & test_episodes) == 0
    print("  No overlap between splits: OK")

    batch = next(iter(dm.train_dataloader()))
    print(f"\n  image:      {batch['obs']['image'].shape}")
    print(f"  agent_pos:  {batch['obs']['agent_pos'].shape}")
    print(f"  keypoint:   {batch['obs']['keypoint'].shape}")
    print(f"  n_contacts: {batch['obs']['n_contacts'].shape}")
    print(f"  action:     {batch['action'].shape}")
    print("\nDataModule ready!")

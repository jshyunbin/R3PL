"""
PushT DataModule 
Loads data from zarr data file.

Borrowing code from diffusion_policy: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/dataset/pusht_dataset.py
"""


from typing import Dict, List, Optional, Tuple
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import copy
import numpy as np
from src.nn.data import ReplayBuffer
from src.nn.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from src.nn.common.normalize_util import get_image_range_normalizer
from src.nn.common.pytorch_util import dict_apply
from src.nn.modules import LinearNormalizer


from src.nn.utils import RankedLogger
from src.nn.utils.constants import IGNORE_LABEL_ID

log = RankedLogger(__name__, rank_zero_only=True)

class PushtDataset(Dataset):
    """Pusht dataset for training and evaluation."""
    def __init__(
            self, 
            zarr_path: str, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=423,
            val_ratio=0.0,
            test_ratio=0.0,
            max_train_episodes=None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action']
            )

        val_mask, test_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
            )
        train_mask = ~(val_mask | test_mask)
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed
            )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
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
            episode_mask=self.val_mask
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
            episode_mask=self.test_mask
            )
        test_set.mask = self.test_mask
        return test_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
    
class PushtDataModule(LightningDataModule):
    """Pusht datamodule for training and evaluation."""
    
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

        self.train_dataset = PushtDataset(
            zarr_path=self.data_dir,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            seed=self.seed,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            max_train_episodes=self.max_train_episodes,
        )
        log.info(f"PushtDataModule: Loaded dataset from {data_dir}")


    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.val_dataset = self.train_dataset.get_validation_dataset()
        
        if stage == 'test' or stage is None:
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
    # Test the datamodule
    print("Testing PushtDataModule...")
    print("="*60)

    dm = PushtDataModule(data_dir="data/pusht/pusht_cchi_v7_replay.zarr", batch_size=4, horizon=16, num_workers=0)
    dm.setup("fit")

    # Verify no overlap
    train_episodes = set(np.where(dm.train_dataset.train_mask)[0])
    val_episodes = set(np.where(dm.train_dataset.val_mask)[0])
    test_episodes = set(np.where(dm.train_dataset.test_mask)[0])

    print("\n✓ Verification:")
    print(f"  Train: {len(train_episodes)} episodes")
    print(f"  Val:   {len(val_episodes)} episodes")
    print(f"  Test:  {len(test_episodes)} episodes")
    print(
        f"  Train ∩ Val: {len(train_episodes & val_episodes)} {'✓' if len(train_episodes & val_episodes) == 0 else '✗'}"
    )
    print(
        f"  Train ∩ Test: {len(train_episodes & test_episodes)} {'✓' if len(train_episodes & test_episodes) == 0 else '✗'}"
    )
    print(
        f"  Val ∩ Test: {len(val_episodes & test_episodes)} {'✓' if len(val_episodes & test_episodes) == 0 else '✗'}"
    )

    train_batch = next(iter(dm.train_dataloader()))
    print(f"\n✓ Train batch shape: {train_batch['obs']['image'].shape=}, {train_batch['obs']['agent_pos'].shape=}, {train_batch['action'].shape=}")

    print("\n✅ DataModule ready!")
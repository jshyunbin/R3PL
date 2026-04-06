"""
Robomimic Image Replay DataModule 
Loads data from zarr data file.

Borrowing code from unified video action: https://github.com/ShuangLI59/unified_video_action/blob/main/unified_video_action/dataset/robomimic_replay_image_dataset.py
"""

from typing import Dict, List, Optional
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.nn.common.pytorch_util import dict_apply
from src.nn.modules import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from src.nn.common.rotation_transformer import RotationTransformer
from src.nn.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from src.nn.data.replay_buffer import ReplayBuffer
from src.nn.common.sampler import SequenceSampler, get_val_mask
from src.nn.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_resnet_imagenet_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

register_codecs()


class RobomimicReplayImageDataset(Dataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        abs_action=False,
        rotation_rep="rotation_6d",  # ignored when abs_action=False
        use_legacy_normalizer=False,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        data_aug=False,
        normalizer_type=None,
    ):

        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + ".zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            print("Cache path:", cache_zarr_path)

            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

    
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask, _ = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

        self.data_aug = data_aug

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

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                # dual arm
                this_normalizer = (
                    robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
                )
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_resnet_imagenet_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            if self.n_obs_steps is None:
                assert np.sum(data[key][T_slice] != data[key]) == 0

            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            # T,C,H,W
            del data[key]

        for key in self.lowdim_keys:
            if self.n_obs_steps is None:
                assert np.sum(data[key][T_slice] != data[key]) == 0

            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file["data"]
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array(
            "episode_ends",
            episode_ends,
            dtype=np.int64,
            compressor=None,
            overwrite=True,
        )

        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["action"]["shape"]
                ), log.warning(f"{this_data.shape=}, {shape_meta["action"]["shape"]=}")
            else:
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["obs"][key]["shape"]
                ), log.warning(f"{this_data.shape=}, {shape_meta["obs"][key]["shape"]=}")
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype,
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
        ) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=this_compressor,
                        dtype=np.uint8,
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f"demo_{episode_idx}"]
                        hdf5_arr = demo["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )

class RobomimicDataModule(LightningDataModule):
    """Robomimic datamodule for training and evaluation."""
    
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        abs_action=False,
        rotation_rep="rotation_6d",  # ignored when abs_action=False
        use_legacy_normalizer=False,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        data_aug=False,
        normalizer_type=None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.horizon = horizon

        self.train_dataset = RobomimicReplayImageDataset(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            n_obs_steps=n_obs_steps,
            abs_action=abs_action,
            rotation_rep=rotation_rep,
            use_legacy_normalizer=use_legacy_normalizer,
            use_cache=use_cache,
            data_aug=data_aug,
            seed=self.seed,
            val_ratio=self.val_ratio,
        )
        log.info(f"RobomimicDataModule: Loaded dataset from {dataset_path}")


    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.val_dataset = self.train_dataset.get_validation_dataset()
        
    
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

if __name__ == "__main__":
    # Test the datamodule
    print("Testing RobomimicDataModule...")
    print("="*60)


    # def print_hdf5_tree(name, obj):
    #     """
    #     Function to print the name and type of an HDF5 object during traversal.
    #     """
    #     # Use 'name' to get the full path, and 'obj' to get the HDF5 object
    #     indent = '  ' * name.count('/')
    #     if isinstance(obj, h5py.Group):
    #         print(f"{indent}📁 {name.split('/')[-1]} (Group)")
    #     elif isinstance(obj, h5py.Dataset):
    #         print(f"{indent}📊 {name.split('/')[-1]} (Dataset: shape {obj.shape}, type {obj.dtype})")
    #     else:
    #         print(f"{indent}📦 {name.split('/')[-1]} (Other)")

    # # Open the HDF5 file in read mode
    # file_name = 'data/stack_color/image256.hdf5'
    # with h5py.File(file_name, 'r') as f:
    #     print(f"--- File structure of '{file_name}' ---")
    #     # Start traversing from the root group ('/')
    #     f.visititems(print_hdf5_tree)

    shape_meta = {
        "image_resolution": 256,
        "action": {
            "shape": [10],
        },
        "obs": {
            "robot0_eef_pos": {
                "shape": [3]
            },
            "robot0_eef_quat": {
                "shape": [4]
            },
            "robot0_eye_in_hand_image": {
                "shape": [3, 256, 256],
                "type": 'rgb'
            },
            "robot0_gripper_qpos": {
                "shape": [2]
            },
            "agentview_image": {
                "shape": [3, 256, 256],
                "type": 'rgb'
            },
        }
    }

    dm = RobomimicDataModule(
        shape_meta=shape_meta,
        dataset_path="data/stack_color/image256.hdf5", 
        batch_size=4,
        horizon=400,
        num_workers=0,
        pad_before=1,
        pad_after=7,
        val_ratio=0.02,
        abs_action=True,
        use_cache=True,
        )
    dm.setup("fit")

    # Verify no overlap
    train_episodes = set(np.where(dm.train_dataset.train_mask)[0])
    val_episodes = set(np.where(dm.train_dataset.val_mask)[0])

    print("\n✓ Verification:")
    print(f"  Train: {len(train_episodes)} episodes")
    print(f"  Val:   {len(val_episodes)} episodes")
    print(
        f"  Train ∩ Val: {len(train_episodes & val_episodes)} {'✓' if len(train_episodes & val_episodes) == 0 else '✗'}"
    )

    train_batch = next(iter(dm.train_dataloader()))
    for key, value in shape_meta['obs'].items():
        print(f"\n✓ {key} observation shape: {train_batch['obs'][key].shape}")
        print(f"observation sample: min: {torch.min(train_batch['obs'][key])}, max: {torch.max(train_batch['obs'][key])}")

    print("\n✅ DataModule ready!")

    import cv2
    out = cv2.VideoWriter('demo/data.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (256, 256))
    for image in train_batch['obs']['agentview_image'][0, ...]:
        image = np.array(image*255, dtype=np.uint8)
        image = np.moveaxis(image, 0, -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)
    out.release()
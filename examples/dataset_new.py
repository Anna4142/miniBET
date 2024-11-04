import abc
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import einops
import numpy as np
import torch
from torch import default_generator, randperm
from itertools import accumulate
from torch.utils.data import Dataset, Subset, TensorDataset


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])


class D4RLDataset(TrajectoryDataset):
    def __init__(self, env_name, dataset_path, future_seq_len, goal_conditional):
        import gym
        import d4rl_pybullet

        self.env = gym.make(env_name)
        # Load dataset from the specified path
        dataset = self.env.get_dataset(h5path=dataset_path)
        observations = dataset['observations']
        actions = dataset['actions']
        terminals = dataset['terminals']

        # Handle the absence of 'timeouts' key
        if 'timeouts' in dataset:
            timeouts = dataset['timeouts']
        else:
            # Create an array of zeros (False) for timeouts
            timeouts = np.zeros_like(terminals, dtype=bool)

        self.observations = torch.from_numpy(observations).float()
        self.actions = torch.from_numpy(actions).float()
        self.terminals = terminals
        self.timeouts = timeouts

        self.future_seq_len = future_seq_len
        self.goal_conditional = goal_conditional

        # Extract goals if present
        if self.goal_conditional:
            if 'goals' in dataset:
                goals = dataset['goals']
            elif 'desired_goal' in dataset:
                goals = dataset['desired_goal']
            else:
                raise ValueError("Dataset does not contain goals, but 'goal_conditional' is set to true.")
            self.goals = torch.from_numpy(goals).float()
        else:
            self.goals = None

        # Split trajectories based on terminals and timeouts
        self.trajectories = []
        trajectory = {'observations': [], 'actions': []}
        for i in range(len(self.observations)):
            trajectory['observations'].append(self.observations[i])
            trajectory['actions'].append(self.actions[i])
            if self.terminals[i] or self.timeouts[i]:
                trajectory['observations'] = torch.stack(trajectory['observations'])
                trajectory['actions'] = torch.stack(trajectory['actions'])
                if self.goal_conditional:
                    trajectory['goal'] = self.goals[i]
                self.trajectories.append(trajectory)
                trajectory = {'observations': [], 'actions': []}
        # Handle the last trajectory if it doesn't end with a terminal or timeout
        if len(trajectory['observations']) > 0:
            trajectory['observations'] = torch.stack(trajectory['observations'])
            trajectory['actions'] = torch.stack(trajectory['actions'])
            if self.goal_conditional:
                trajectory['goal'] = self.goals[-1]  # Assuming the last goal
            self.trajectories.append(trajectory)

    def __len__(self):
        return len(self.trajectories)

    def get_seq_length(self, idx):
        return len(self.trajectories[idx]['observations'])

    def __getitem__(self, idx):
        observations = self.trajectories[idx]['observations']
        actions = self.trajectories[idx]['actions']
        mask = torch.ones(len(observations), dtype=torch.float32)
        if self.goal_conditional:
            goal = self.trajectories[idx]['goal']
            return observations, actions, mask, goal
        else:
            return observations, actions, mask



class TrajectorySlicerDataset(TrajectoryDataset):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
    ):
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.slices = []
        min_seq_length = np.inf
        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                self.slices += [
                    (i, start, start + window) for start in range(T - window)
                ]

        if min_seq_length < window:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        data = self.dataset[i]
        observations, actions, mask = data[0], data[1], data[2]
        obs_slice = observations[start:end]
        act_slice = actions[start:end]
        mask_slice = mask[start:end]

        if self.future_conditional:
            if len(data) < 4:
                raise ValueError("Dataset does not contain goals, but 'future_conditional' is set to true.")
            goal = data[3]  # Extract goal
            return obs_slice, act_slice, goal, mask_slice
        else:
            return obs_slice, act_slice, mask_slice

def get_train_val_sliced(
    traj_dataset: TrajectoryDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    window_size: int = 10,
    future_conditional: bool = False,
    min_future_sep: int = 0,
    future_seq_len: Optional[int] = None,
    only_sample_tail: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    traj_slicer_kwargs = {
        "window": window_size,
        "future_conditional": future_conditional,
        "min_future_sep": min_future_sep,
        "future_seq_len": future_seq_len,
        "only_sample_tail": only_sample_tail,
        "transform": transform,
    }
    train_slices = TrajectorySlicerDataset(train, **traj_slicer_kwargs)
    val_slices = TrajectorySlicerDataset(val, **traj_slicer_kwargs)
    return train_slices, val_slices


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    offsets = list(accumulate(lengths))
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(offsets, lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def get_d4rl_dataset(
    env_name,
    dataset_path,
    future_seq_len,
    goal_conditional,
    train_fraction=0.9,
    random_seed=42,
    window_size=10,
    only_sample_tail: bool = False,
    min_future_sep: int = 0,
    transform: Optional[Callable[[Any], Any]] = None,
):
    traj_dataset = D4RLDataset(
        env_name=env_name,
        dataset_path=dataset_path,
        future_seq_len=future_seq_len,
        goal_conditional=goal_conditional,
    )

    return get_train_val_sliced(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
        window_size=window_size,
        only_sample_tail=only_sample_tail,
        future_conditional=goal_conditional,
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
    )

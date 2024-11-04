# goal_functions.py

import numpy as np
from typing import Optional, Tuple

class GoalFunction:
    def __init__(self, goal_dim: int, method: str = "future", noise_std: float = 0.1):
        """
        Initializes the GoalFunction.

        Args:
            goal_dim (int): Dimension of the goal.
            method (str): Method to generate goals. Options include 'future', 'random', 'fixed'.
            noise_std (float): Standard deviation of the noise added to the goal (used in 'future' method).
        """
        self.goal_dim = goal_dim
        self.method = method
        self.noise_std = noise_std

    def __call__(self, env, current_obs: np.ndarray, eval_idx: int, step: int) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Generates a goal based on the current observation.

        Args:
            env: The environment.
            current_obs (np.ndarray): The current observation from the environment.
            eval_idx (int): Evaluation index.
            step (int): Current step in the episode.

        Returns:
            Tuple containing the goal and any additional info.
        """
        if self.method == "future":
            # Example: Add Gaussian noise to the current observation to create a goal
            goal = current_obs + np.random.normal(0, self.noise_std, size=self.goal_dim)
        elif self.method == "random":
            # Generate a random goal within the observation space
            goal = env.observation_space.sample()
        elif self.method == "fixed":
            # Set a fixed goal (e.g., origin)
            goal = np.zeros(self.goal_dim)
        else:
            raise ValueError(f"Unknown goal generation method: {self.method}")

        return goal, None  # Additional info can be added if needed

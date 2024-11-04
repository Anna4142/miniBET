# train_new.py

import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

import wandb

import dataset_new
# Removed GoalFunction import since it's no longer needed
# from goal_functions import GoalFunction  # Ensure this import is removed

# Ensure MUJOCO_GL is set for rendering (optional)
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

def seed_everything(random_seed: int):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

@hydra.main(config_path=".", config_name="train_new", version_base="1.2")
def main(cfg):
    # Print the resolved configuration
    print(OmegaConf.to_yaml(cfg))

    # Set random seeds for reproducibility
    seed_everything(cfg.seed)

    # Instantiate the dataset
    train_data, test_data = hydra.utils.instantiate(cfg.data)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=True
    )

    # Instantiate the model and move it to the configured device
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)

    # Load model weights if a load path is provided
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))

    # Configure the optimizer
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )

    # Removed goal_fn instantiation since it's no longer needed
    # goal_fn = hydra.utils.instantiate(cfg.goal_fn)
    # if goal_fn is None:
    #     print("Goal function is None. Running in non-goal-conditioned mode.")
    # else:
    #     print("Goal function instantiated. Running in goal-conditioned mode.")

    # Instantiate the environment
    env = hydra.utils.instantiate(cfg.env.gym)

    # Initialize Weights & Biases (uncomment if using wandb)
    #run = wandb.init(
       # project=cfg.wandb.project,
       # entity=cfg.wandb.entity,
       # config=OmegaConf.to_container(cfg, resolve=True),
    #)
    #run_name = run.name or "Offline"
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def eval_on_env(cfg, num_evals=cfg.num_env_evals, num_eval_per_goal=1):
        """
        Evaluates the model in the environment.

        Args:
            cfg: Configuration object.
            num_evals: Number of evaluation episodes.
            num_eval_per_goal: Number of evaluations per goal.

        Returns:
            Average reward over all evaluations.
        """
        avg_reward = 0
        for eval_idx in range(num_evals):
            for _ in range(num_eval_per_goal):
                obs_stack = deque(maxlen=cfg.eval_window_size)
                obs = env.reset()
                obs_stack.append(obs)
                done, step, total_reward = False, 0, 0

                # No goal generation since goal conditioning is disabled
                goal = None

                while not done:
                    # Convert observation stack to tensor
                    obs_tensor = torch.from_numpy(np.stack(obs_stack)).float().to(cfg.device)

                    # No goal tensor
                    goal_tensor = None

                    # Get action from the model
                    action, _, _ = cbet_model(
                        obs_tensor.unsqueeze(0),
                        goal_tensor,  # Pass None for non-goal-conditioned mode
                        None,
                    )

                    # Take a step in the environment
                    action_np = action[0, -1, :].cpu().numpy()
                    obs_new, reward, done, info = env.step(action_np)
                    step += 1
                    total_reward += reward
                    obs_stack.append(obs_new)

                avg_reward += total_reward
        return avg_reward / (num_evals * num_eval_per_goal)

    # Training loop
    for epoch in tqdm.trange(cfg.epochs, desc="Epochs"):
        cbet_model.eval()

        # Evaluate on the environment at specified frequency
        if epoch % cfg.eval_on_env_freq == 0:
            avg_reward = eval_on_env(cfg)
            #wandb.log({"eval_on_env": avg_reward})
            #print(f"Epoch {epoch}: Average Reward = {avg_reward}")

        # Evaluate on the test dataset at specified frequency
        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            cbet_model.eval()
            with torch.no_grad():
                for data in test_loader:
                    # Non-goal-conditioned mode: data includes obs, act, mask
                    obs, act, mask = data[0], data[1], data[2]
                    goal = None  # No goal in non-goal-conditioned mode

                    # Forward pass
                    _, loss, loss_dict = cbet_model(obs, goal, act)
                    total_loss += loss.item()

                    # Log loss components
                    #wandb.log({"eval/{}".format(k): v for k, v in loss_dict.items()})

            avg_test_loss = total_loss / len(test_loader)
            print(f"Epoch {epoch}: Test Loss = {avg_test_loss}")

        # Training phase
        cbet_model.train()
        for data in tqdm.tqdm(train_loader, desc="Training Batches"):
            optimizer.zero_grad()

            # Non-goal-conditioned mode: data includes obs, act, mask
            obs, act, mask = data[0], data[1], data[2]
            goal = None  # No goal in non-goal-conditioned mode

            # Forward pass
            _, loss, loss_dict = cbet_model(obs, goal, act)

            # Log loss components
            #wandb.log({"train/{}".format(k): v for k, v in loss_dict.items()})

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Save the model at specified frequency
        if epoch % cfg.save_every == 0:
            cbet_model.save_model(save_path)
            print(f"Epoch {epoch}: Model saved at {save_path}")

    # Final evaluation after training
    avg_reward = eval_on_env(
        cfg,
        num_evals=cfg.num_final_evals,
        num_eval_per_goal=cfg.num_final_eval_per_goal,  # Consider renaming or removing
    )
    #wandb.log({"final_eval_on_env": avg_reward})
    print(f"Final Evaluation: Average Reward = {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    main()

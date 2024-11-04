import gym
import mujoco
import d4rl


def test_kitchen():
    print("Testing Franka Kitchen Environment...")

    try:
        # Create kitchen environment
        env = gym.make('kitchen-mixed-v0')
        print("✓ Environment created successfully")

        # Get initial observation
        obs = env.reset()
        print("✓ Environment reset successful")
        print(f"Observation space shape: {obs.shape}")

        # Try a few steps
        for i in range(100):
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
            env.render()  # This will show the visualization

            if done:
                env.reset()

        print("✓ Successfully ran simulation steps")

        # Get dataset info
        dataset = env.get_dataset()
        print("\nDataset information:")
        print(f"- Number of transitions: {len(dataset['observations'])}")
        print(f"- Observation shape: {dataset['observations'].shape}")
        print(f"- Action shape: {dataset['actions'].shape}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")

    finally:
        env.close()


if __name__ == "__main__":
    test_kitchen()
import gym
import d4rl_pybullet

# List of available d4rl_pybullet environments
env_ids = [
    'hopper-bullet-random-v0',
    'hopper-bullet-medium-v0',
    'hopper-bullet-mixed-v0',
    'halfcheetah-bullet-random-v0',
    'halfcheetah-bullet-medium-v0',
    'halfcheetah-bullet-mixed-v0',
    'ant-bullet-random-v0',
    'ant-bullet-medium-v0',
    'ant-bullet-mixed-v0',
    'walker2d-bullet-random-v0',
    'walker2d-bullet-medium-v0',
    'walker2d-bullet-mixed-v0'
]

# Loop through all environments to download the datasets
for env_id in env_ids:
    env = gym.make(env_id)
    dataset = env.get_dataset()
    print(f"Downloaded dataset for: {env_id}")

print("All datasets have been downloaded.")


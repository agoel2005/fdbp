import sys
import os

# Add the VLM2Vec directory to the Python path
print(sys.path)
sys.path.append(os.path.join(os.path.dirname(__file__), 'VLM2Vec'))
print(sys.path)


import gymnasium as gym
import numpy as np
import heapq
import time
import logging
from tqdm import tqdm
from minigrid.wrappers import FullyObsWrapper
import pandas as pd
from vlmagent import VLMAgent, AStarAgent

# Set up logging
logging.basicConfig(filename='agent_training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Environment setup
env_human = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")) #"MiniGrid-Fetch-8x8-N3-v0"
env_array = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")) #"MiniGrid-Fetch-8x8-N3-v0"

observation, info = env_human.reset(seed=42)
env_array.reset(seed=42)


def test_agent_performance(agent, env, num_trials=5):
    """
    Function to test the agent and return the average number of steps to reach the goal using top-k plans.
    :param agent: AStarAgent instance.
    :param env: Environment instance to test the agent.
    :param num_trials: Number of trials to run for averaging.
    :return: Average number of steps to reach the goal.
    """
    total_steps = 0
    # Get top-k plans for the trial
    init_observation, info = env.reset(seed=42)
    #start_state = agent.feature_extractor(torch.tensor(init_observation['image'], dtype=torch.float32).unsqueeze(0))
    top_k_plans = agent.get_top_k_plans(env, k=num_trials)
    if len(top_k_plans) == 0:
        return None
    for i in range(num_trials):
        observation, info = env.reset(seed=42)
        assert np.all(init_observation['image'] == observation['image'])
        terminated = False
        steps = 0

        # Use the best plan to execute in the environment
        best_plan, _ = top_k_plans[0]
        for state_tuple, action in best_plan[1:]:  # Skip the initial state
            if action is None:
                continue
            if terminated or steps >= 15:
                break
            observation, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated:
                logger.info("[SUCCESS] Goal reached during testing.")
                #agent.generate_states_via_search("done.csv")
                break
        total_steps += steps
    return total_steps / num_trials

print('hi')
feature_extractor = VLMAgent()
print('hello')
goal_state = feature_extractor.get_goal_state(env_array.render())
print('hola')
agent = AStarAgent(feature_extractor, goal_state)
print('done')


# Test the agent on the environment
test_observation, info = env_human.reset(seed=42)
env_array.reset(seed=42)
terminated = False
while not terminated:
    action = agent.get_action(env_array)
    print(action)
    test_observation, reward, terminated, truncated, _ = env_human.step(action)
    env_array.step(action)
    if terminated or truncated:
        logger.info("Goal reached during testing.")
        break

env_human.close()
env_array.close()
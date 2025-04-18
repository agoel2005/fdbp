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
import torch

# Set up logging
logging.basicConfig(filename='agent_training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Environment setup
env = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")) #"MiniGrid-Fetch-8x8-N3-v0"
#env_array = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")) #"MiniGrid-Fetch-8x8-N3-v0"

observation, info = env.reset(seed=42)
#env_array.reset(seed=42)
import time
import random
for i in range(100):
    action = int(random.random()*3)
    env.step(action)
    print(env.agent_pos, env.agent_dir)
    time.sleep(1)


#right: 0, down: 1, left: 2, up: 3
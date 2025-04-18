import sys
import os

# Add the VLM2Vec directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'VLM2Vec'))


import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from vlmagent import VLMAgent, AStarAgent
import torch
from collections import deque


# Environment setup
env_human = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")) #"MiniGrid-Fetch-8x8-N3-v0"
env = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")) #"MiniGrid-Fetch-8x8-N3-v0"


observation, info = env_human.reset(seed=42)
env.reset(seed=42)

feature_extractor = VLMAgent()
rendered_image = env.render()


goal_state = feature_extractor.get_goal_state(rendered_image)
torch.save(goal_state, 'predicted_goal.pt')


# Reset initial state
obs, _ = env.reset()


initial_state = (env.agent_pos[0], env.agent_pos[1], env.agent_dir)
GOAL_POS = (2, 2)
DIRECTIONS = 4

# Track visited states and how to get to them
visited = set()
visited.add(initial_state)
paths_to_states = {initial_state: []}

# BFS queue: (action history)
queue = deque()
queue.append([])

def get_state(e):
    return (e.agent_pos[0], e.agent_pos[1], e.agent_dir)

# Explore state space
while queue and len(visited) < 33:
    history = queue.popleft()

    # Replay history to get to the right state
    env_copy = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode=None))
    env_copy.reset()
    for a in history:
        env_copy.step(a)

    for action in [0, 1, 2]:  # left, right, forward
        env_clone = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode=None))
        env_clone.reset()
        for a in history:
            env_clone.step(a)
        env_clone.step(action)
        new_state = get_state(env_clone)

        if new_state in visited:
            # Skip repeated orientation in goal cell
            if new_state[:2] == GOAL_POS:
                continue
            else:
                continue

        visited.add(new_state)
        paths_to_states[new_state] = history + [action]
        queue.append(history + [action])

print(f"Found {len(paths_to_states)} unique states!")

# Replay all states visually
for i, (state, actions) in enumerate(paths_to_states.items()):
    env.reset()
    for a in actions:
        env.step(a)
    
    for a in range(3):
        #next_state = feature_extractor.evaluate(env.render(), action)
        next_state = feature_extractor.evaluate_text(action, env.agent_pos, env.agent_dir)
        torch.save(next_state, f'text_tensors/pos={state[:2]}, dir={state[2]}, action={a}.pt')
    print(f"[{i+1}/33] State: pos={state[:2]}, dir={state[2]}")


env_human.close()
env.close()
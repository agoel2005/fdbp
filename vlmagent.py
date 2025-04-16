from VLM2Vec.src.model import MMEBModel
from VLM2Vec.src.arguments import ModelArguments
from VLM2Vec.src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens

import torch
from transformers import HfArgumentParser, AutoProcessor
from PIL import Image
import numpy as np
import pandas as pd 
import gymnasium as gym
import numpy as np
import heapq
import time
import logging
from tqdm import tqdm
from minigrid.wrappers import FullyObsWrapper
import pandas as pd
from PIL import Image

class VLMAgent:
    def __init__(self):
        self.model_args = ModelArguments(
            model_name='Qwen/Qwen2-VL-2B-Instruct',  # Can remain if used just for metadata
            checkpoint_path='./vlm2vec_qwen2vl_2b',  # Local path to downloaded model
            pooling='last',
            normalize=True,
            model_backbone='qwen2_vl',
            lora=True
        )

        # Load processor and model from local path
        self.processor = load_processor(self.model_args)
        self.model = MMEBModel.load(self.model_args)

        # Send to GPU with bf16
        self.model = self.model.to('cuda', dtype=torch.bfloat16)
        self.model.eval()
    
    def extract_state(self, image):
        prompt="You are in a minigrid environment. You are given an image of what the environment looks like. Describe the environment in context to how far the player is from the goal state."
        inputs = self.processor(text=prompt,
                        images=Image.fromarray(image),
                        return_tensors="pt")
        inputs = {key: value.to('mps') for key, value in inputs.items()}
        qry_output = self.model(qry=inputs)["qry_reps"]
        return qry_output 
    
    def evaluate(self,image, action):
        
        # Image + Text -> Text
        prompt=f"""You are in a minigrid environment. Given the current environment shown in the image
        and the action {action}, I want you to answer what happens in the environment after this action is taken"""

        
        inputs = self.processor(text=prompt,
                        images=Image.fromarray(image),
                        return_tensors="pt")
        inputs = {key: value.to('mps') for key, value in inputs.items()}
        qry_output = self.model(qry=inputs)["qry_reps"]
        return qry_output 

    def get_goal_state(self, image):
         # Image + Text -> Text
        prompt=f"""You are in a minigrid environment. Given the current environment shown in the image
        ,I want you to answer what the environment looks like when the agent reaches the goal state."""

        inputs = self.processor(text=prompt,
                        images=Image.fromarray(image),
                        return_tensors="pt")
        inputs = {key: value.to('mps') for key, value in inputs.items()}
        qry_output = self.model(qry=inputs)["qry_reps"]
        return qry_output 

    def compute_similarity(self, curr, goal):
        return self.model.compute_similarity(curr, goal)



# A* Planning Agent
class AStarAgent:
    def __init__(self, feature_extractor, goal_state):
        self.feature_extractor = feature_extractor
        self.goal_state = goal_state
        self.visited_nodes = []  # To store visited nodes for EPT Loss calculation
        self.max_planning_time = 0.05 # Maximum planning time in seconds
        self.atol = 5e-4 # Absolute tolerance for goal state

    def heuristic(self, state):
        return torch.norm(state - self.goal_state, p=2).item()

    def tensor_to_tuple(self, tensor):
        return tuple(tensor.detach().cpu().numpy().flatten())

    def get_action(self, env):
        curr_state = env.render()
        min_distance = 1
        best_action = -1
        for action in range(env.action_space.n):
            next_state = self.feature_extractor.evaluate(curr_state, action)
            dis = self.feature_extractor.compute_similarity(next_state, self.goal_state)
            if(dis < min_distance):
                min_distance = dis 
                best_action = action 

        return best_action 




    """
    
    def generate_states(self, file_path="graph.csv"):
        inputs = {}
        # Initialize the base array
        base_array = np.array([[[2, 5, 0]] * 5] * 5, dtype=np.uint8)
        
        # Add inner 3x3 block with the default values [1, 0, 0]
        for i in range(1, 4):
            for j in range(1, 4):
                if i != 3 or j != 3:
                    base_array[i][j] = [1, 0, 0]
                else:
                    base_array[i][j] = [8, 1, 0]
        
        # Loop over each position in the 3x3 block (positions [1,1] to [3,3])
        for i in range(1, 4):
            for j in range(1, 4):
                for k in range(0, 4):
                    # Create a copy of the base array
                    modified_array = np.copy(base_array)
                    
                    # Swap the value at position (i, j) in the inner block
                    modified_array[i][j] = [10, k, 0]
                    
                    # Print the resulting array
                    inputs[f"{i}_{j}_{k}"] = modified_array

        # Initialize a list to store all states and a list for the corresponding keys
        states = []
        keys = []

        # Extract states from the inputs and store them with corresponding keys
        for key, val in inputs.items():
            state = self.feature_extractor(torch.tensor(val, dtype=torch.float32).unsqueeze(0))
            states.append(state)
            keys.append(key)
            for action in range(7):
                state = self.model(state, action)
                states.append(state)
                keys.append(key + f"_{action}")

        # Stack all states into a single tensor for pairwise distance computation
        states_tensor = torch.cat(states, dim=0)  # Shape: (N, feature_dim), where N is the number of states

        # Compute pairwise distances between all states
        pairwise_distances = torch.cdist(states_tensor, states_tensor)

        # Convert pairwise distances to a Pandas DataFrame with the keys as index and column names
        distance_df = pd.DataFrame(pairwise_distances.detach().numpy(), index=keys, columns=keys)

        # Display the DataFrame
        distance_df.to_csv(file_path)
    """
    """
    
    def generate_states_via_search(self, file_path="graph.csv"):
        states = []
        keys = []

        start_state = self.feature_extractor(torch.tensor(demo_buffer[0][0], dtype=torch.float32).unsqueeze(0))
        states.append(start_state)
        keys.append("start")
        states.append(self.goal_state)
        keys.append("goal")
        plans = self.a_star_search(start_state, verbose=True)
        # Extract states from the inputs and store them with corresponding keys
        i = 0
        for key, val in plans.items():
            state = torch.tensor(key, dtype=torch.float32).unsqueeze(0)
            states.append(state)
            keys.append(str(i))
            i += 1
            if val is None:
                continue
            from_state, action = val
            from_state = torch.tensor(from_state, dtype=torch.float32).unsqueeze(0)
            states.append(from_state)
            keys.append(str(i))
            states.append(state)
            keys.append(f"{i}_{action}")
            i += 1

        # Stack all states into a single tensor for pairwise distance computation
        states_tensor = torch.cat(states, dim=0)  # Shape: (N, feature_dim), where N is the number of states

        # Convert pairwise distances to a Pandas DataFrame with the keys as index and column names
        distance_df = pd.DataFrame(states_tensor.detach().numpy(), index=keys)

        # Display the DataFrame
        distance_df.to_csv(file_path)
        """

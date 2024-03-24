import numpy as np
from ollama import Client
import re
import llm.prompts as prompts
from llm.prompts import DECODE_MODE_TUPLE, DECODE_MODE_LIST
import logging

class Nashor:
    """
    Nashor agent that uses Llama to generate actions based on the observation and action mask.
    """

    def __init__(self, map_size=8):
        self.rows_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        self.action_types = ["NOOP", "move", "harvest", "return", "produce", "attack"]
        self.location_parameters = ["north", "east", "south", "west"]
        self.unit_types = ["resource", "base", "barracks", "worker", "light", "heavy", "ranged"]
        self.owners = ["NEUTRAL", "PLAYER", "ENEMY"]
        self.obs_unit_types = ["NO_UNIT"] + self.unit_types

        self.map_size = map_size
        self.positions = self._create_positions()
        self.client = Client(host='http://localhost:11434')

        self.attack_range = 7
        self.half_attack_range = self.attack_range//2

        file_handler = logging.FileHandler('nashor.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.addHandler(file_handler)

    def _create_positions(self):
        """Create a 3D array with the positions of the map."""
        positions = np.array([
            [
                [f"{self.rows_letters[i]}{j}"] for j in range(self.map_size)
            ] for i in range(self.map_size)
        ])
        return positions
    
    def _encode_action_mask_tile(self, action_mask_tiles, row, col):
        """Convert the action mask for a particular tile to a list of strings with the valid actions."""

        resulting_strings = []

        unit = f"{self.rows_letters[row]}{col}"

        # NOOP
        if action_mask_tiles[0] == 1:
            resulting_strings.append(f"({unit}, NOOP)")

        # move
        if action_mask_tiles[1] == 1:
            for i, param in zip(action_mask_tiles[6:10], self.location_parameters):
                if i == 1:
                    resulting_strings.append(f"({unit}, move, {param})")

        # harvest
        if action_mask_tiles[2] == 1:
            for i, param in zip(action_mask_tiles[10:14], self.location_parameters):
                if i == 1:
                    resulting_strings.append(f"({unit}, harvest, {param})")

        # return
        if action_mask_tiles[3] == 1:
            for i, param in zip(action_mask_tiles[14:18], self.location_parameters):
                if i == 1:
                    resulting_strings.append(f"({unit}, return, {param})")

        # produce
        if action_mask_tiles[4] == 1:
            for i, param in zip(action_mask_tiles[18:22], self.location_parameters):
                if i == 1:
                    for j, param2 in zip(action_mask_tiles[22:29], self.unit_types):
                        if j == 1:
                            resulting_strings.append(f"({unit}, produce, {param}, {param2})")

        # attack
        if action_mask_tiles[5] == 1:
            mask_atk = action_mask_tiles[29:].reshape(7,7)
            
            for atk_row_rel in range(self.attack_range):
                for atk_col_rel in range(self.attack_range):
                    if mask_atk[atk_row_rel][atk_col_rel] == 1:
                        atk_row_abs = row + (atk_row_rel - self.half_attack_range)
                        atk_col_abs = col + (atk_col_rel - self.half_attack_range)
                        atk_unit = f"{self.rows_letters[atk_row_abs]}{atk_col_abs}"
                        resulting_strings.append(f"({unit}, attack, {atk_unit})")

        return resulting_strings

    def _encode_action_mask(self, action_mask):
        """Encode the action mask to a prompt string."""

        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        action_mask = action_mask.reshape(self.map_size, self.map_size, -1)
        
        valid_actions = []
        for row in range(self.map_size):
            for col in range(self.map_size):
                am_tile = action_mask[row][col]
                valid_action = self._encode_action_mask_tile(am_tile, row, col)
                if len(valid_action) > 0:
                    valid_actions.extend(valid_action)

        prompt_valid_actions = "\n".join(valid_actions)

        return prompt_valid_actions

    def _encode_observation_features(self, features):
        """Encode the observation features to a string."""

        feat_hit_points = features[0:5]
        hit_points = np.argmax(feat_hit_points)

        feat_resources = features[5:10]
        resources = np.argmax(feat_resources)
        resources = '-' if resources == 0 else f"resources: {resources}"

        feat_owner = features[10:13]
        owner = np.argmax(feat_owner)
        str_owner = self.owners[owner]

        feat_unit_types = features[13:21]
        unit_type = np.argmax(feat_unit_types)
        str_unit_type = self.obs_unit_types[unit_type]

        feat_action_type = features[21:27]
        action_type = np.argmax(feat_action_type)
        str_action_type = self.action_types[action_type]

        obs_str = (
            str_unit_type, 
            f"hp:{hit_points}", 
            resources, 
            str_owner, 
            str_action_type
        )

        return obs_str

    def _encode_observation(self, observation):
        """Encode the observation to a prompt string."""

        encoded_observation = np.apply_along_axis(self._encode_observation_features, 2, observation[0])
        encoded_observation = np.concatenate((self.positions, encoded_observation), axis=-1)
        obs_with_units = encoded_observation[encoded_observation[:, :, 1] != 'NO_UNIT']
        prompt_str = np.apply_along_axis(lambda x: "("+", ".join(x)+")", 1, obs_with_units)
        prompt_str = "\n".join(prompt_str)

        return prompt_str

    def _diff_relative_attack(self, own_unit, atk_unit):
        """Calculate the relative position to the target unit"""
        a_row = self.rows_letters.index(own_unit[0])
        a_col = int(own_unit[1])

        b_row = self.rows_letters.index(atk_unit[0])
        b_col = int(atk_unit[1])

        row_diff = b_row - a_row
        col_diff = b_col - a_col

        atk_linear = ((self.half_attack_range + row_diff) * self.attack_range) + (self.half_attack_range + col_diff) + 1

        return atk_linear
    
    def _decode_string_to_action(self, response):
        """Decode the response string to an action."""

        actions = np.zeros((self.map_size, self.map_size, 7))

        for value in response:
            try:
                s_action = value.strip("()").split(", ")

                unit = s_action[0]
                actiontype_choice = s_action[1]
                actiontype_param = self.action_types.index(actiontype_choice)

                location_choice = s_action[2]
                location_param = self.location_parameters.index(location_choice)

                if len(s_action) > 3:
                    unit_type_choice = s_action[3] 
                    unit_types_param = self.unit_types.index(unit_type_choice)

                    target_unit = s_action[3]
                    attack_param = self._diff_relative_attack(unit, target_unit)

                else:
                    unit_types_param = 0
                    attack_param = 0

                row = self.rows_letters.index(unit[0])
                col = int(unit[1])

                action = [
                    actiontype_param,
                    location_param,
                    location_param,
                    location_param,
                    location_param,
                    unit_types_param,
                    attack_param
                ]

                actions[row][col] = action
            except:
                #print(f"Invalid action: {value}")
                self.logger.warning(f"Invalid action: {value}")
                continue
            #print(s_action)
            

        return actions

    def _decode_action(self, content, decode_mode=DECODE_MODE_TUPLE, prompt_valid_actions=None):
        """Decode the response string to an action."""

        if decode_mode == DECODE_MODE_LIST and prompt_valid_actions is not None:
            found_idx = re.findall(r'\d+', content)

            valid_actions = prompt_valid_actions.split("\n")

            found_actions = []
            for idx in found_idx:
                idx = int(idx)
                if idx < len(valid_actions):
                    found_actions.append(valid_actions[idx])
                else:
                    # print(f"Invalid action index: {idx}")
                    continue

        if decode_mode == DECODE_MODE_TUPLE:
            found_actions = re.findall(r'\((.*?)\)', content)

        self.logger.debug(f"Chosen Actions:\n{found_actions}")
        action = self._decode_string_to_action(found_actions)
        
        return action
    
    def get_action(self, action_mask, observation):
        """Get the next action based on the observation and action mask."""
        
        self.logger.debug("-"*120)
        self.logger.info("Getting action from Nashor")

        # Encode observation and action mask to prompt string to be llm input
        prompt_valid_actions = self._encode_action_mask(action_mask)
        prompt_observation = self._encode_observation(observation)

        # Decide which prompt to use
        prompt, decode_mode = prompts.list_version2(prompt_observation, prompt_valid_actions)
        self.logger.debug(f"Prompt:\n'''\n{prompt}\n'''")

        # Send prompt to Llama
        llama_response = self.client.chat(model='mistral', messages=[
            #{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ])
        self.logger.debug(f"Response:\n'''\n{llama_response}\n'''")

        # Process response based on decode mode
        content = llama_response['message']['content']        
        action = self._decode_action(content, decode_mode=decode_mode, prompt_valid_actions=prompt_valid_actions)

        return action
    

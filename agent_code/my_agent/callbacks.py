import os
import pickle
import random

import numpy as np

###___SET UP___: 
"""
Fixed quantities used only in callbacks.py are initialized here
"""

##__GLOBAL VARS__:

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N = 6 #number of features
epsilon = 0.1

ACTIONS_minus_bomb = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        weights = np.random.rand(N, len(ACTIONS))
        self.model = weights


    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    #!!!!!check which actions are possible


    if self.train:
        if random.random() < epsilon:
            self.logger.debug("Choosing random action (Exploring).")
            return np.random.choice(ACTIONS) #<------------------------- should only contain possible actions
        
        else:
            self.logger.debug("Choosing argmax action (Exploiting).")
            argmax_index = (np.matmul(state_to_features(game_state),self.model)).argmax(axis=0)

            self.logger.debug(f'Agent took action: {ACTIONS[argmax_index]}')
            self.logger.debug(f"Current betas: {self.model}")
            return ACTIONS[argmax_index]

    else:
        self.logger.debug("Choosing argmax action for trained agent.")
        argmax_index = (np.matmul(state_to_features(game_state),self.model)).argmax(axis=0)

        self.logger.debug(f'Agent took action: {ACTIONS[argmax_index]}')
        return ACTIONS[argmax_index]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    features = np.ones(N)

    #get wall values
    
    #up
    features[0] = game_state["field"][game_state["self"][3][0]][game_state["self"][3][1]+1] + 2
    #rihgt
    features[1] = game_state["field"][game_state["self"][3][0]+1][game_state["self"][3][1]] + 2
    #down
    features[2] = game_state["field"][game_state["self"][3][0]][game_state["self"][3][1]-1] + 2
    #left
    features[3] = game_state["field"][game_state["self"][3][0]-1][game_state["self"][3][1]] + 2


    #bomb action possible?
    if game_state["self"][2]:
        features[4] = 1
    else:
        features[4] = 2

    
    #distant to bomb
    #game_state["bombs"][0][0] - game_state["self"][3]

    # # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(np.ones(N))
    
    
    # # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels) #<------------------------------------- is not a stack!! alternatively insert  
    #         #np.concatenate([np_array_1s, np_array_9s])

    # # and return them as a vector
    # return stacked_channels.reshape(-1)

    return features

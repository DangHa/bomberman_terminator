import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = ['F1', 'F2', 'F3', 'F4'] #Replace with actual feature names


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.weights = np.random.rand(len(ACTIONS),len(FEATURES))
        self.q_values = np.zeros(len(ACTIONS))
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
    #Calculate Q
    self.Q = q_function(self, game_state, self.weights) #Does it make sense to store Q here? -> Look at rain.game_events_occurred() to figure out.

    if self.train and random.random() < self.epsilon:
        self.logger.debug("Training Mode (Exploration): Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) #CHANGE THESE?
    else:
        self.logger.debug("Training Mode (Exploitation) / Playing: Choosing action based on max Q.")
        action_index = np.argmax(self.Q)
        return ACTIONS[action_index]


def state_to_features(game_state: dict) -> np.array:
    """
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

    features = np.zeros(len(FEATURES))
    features[0] = 0 #E.g.: game_state['coins'][0]+game_state['self'][3][0] <- just random numbers as example
    features[1] = 0
    return features

def q_function(self, game_state: dict, weights) -> np.array:

    """
    Calculates Q-value of linear regression

    :param game_state: A dictionary describing the current game board.
    :param weights: A numpy array of weigh vectors used for the linear regression (one for each action).
    :return: np.array 
           
    """

    features = state_to_features(game_state)
    self.logger.info("Calculating q-function values.")
    Q = [np.sum(features*weights[i]) for i in range(len(ACTIONS))]

    return np.array(Q)





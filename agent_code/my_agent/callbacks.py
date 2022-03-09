import os
import pickle
import random

import numpy as np

from events import OPPONENT_ELIMINATED


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = ['avoid_wall', 'find_coin', 'find_crate', 'bomb_crate']  #Add feature names


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
        self.weights = np.random.rand(len(FEATURES))
        self.q_values = np.zeros(len(ACTIONS))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.weights = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    #Calculate Q
    self.q_values = q_function(self, game_state, self.weights) #Does it make sense to store Q here? -> Look at rain.game_events_occurred() to figure out.

    if self.train and random.random() < self.epsilon:
        self.logger.debug("Training Mode (Exploration): Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0]) #PHASE 1 (No crates yet)
    else:
        self.logger.debug("Training Mode (Exploitation) / Playing: Choosing action based on max Q.")
        action_index = np.argmax(self.q_values)
        return ACTIONS[action_index]


def state_to_features(self,game_state: dict) -> np.array:
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
        self.logger.debug("Game_state is None.")
        return None

    else: 
        #Extracting relevant game_state values
        agent_coord_x = game_state['self'][3][0]
        agent_coord_y = game_state['self'][3][1]        

        find_coin = find_coins([agent_coord_x, agent_coord_y], game_state)
        avoid_wall = avoid_hitting_wall(game_state)

        # have try yet --> need avoid bombs to test
        find_crate = find_closest_crates([agent_coord_x, agent_coord_y], game_state)
        destroy_crate = bomb_crate([agent_coord_x, agent_coord_y], game_state)

        return np.array([avoid_wall, find_coin, find_crate, destroy_crate])

def q_function(self, game_state: dict, weights) -> np.array:

    """
    Calculates Q-value of linear regression

    :param game_state: A dictionary describing the current game board.
    :param weights: A numpy array of weigh vectors used for the linear regression (one for each action).
    :return: np.array 
           
    """

    features = state_to_features(self,game_state)
    self.logger.info("Calculating q-function values.")
    Q = np.sum([features[i]*self.weights[i] for i in range(len(FEATURES))], axis=0)

    return np.array(Q)

def avoid_hitting_wall(game_state):
    #Extracting relevant game_state values
    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #Engineering features
    features = np.zeros(len(ACTIONS))
    features[0] = game_state['field'][agent_coord_x][agent_coord_y-1]
    features[1] = game_state['field'][agent_coord_x+1][agent_coord_y]
    features[2] = game_state['field'][agent_coord_x][agent_coord_y+1]
    features[3] = game_state['field'][agent_coord_x-1][agent_coord_y]

    return features

def find_coins(agent_location, game_state):
    coin_locations = game_state['coins']

    features = np.zeros(len(ACTIONS))
    closest_coin = None
    closest_dist = 100

    # find the closest coin
    for coin_x, coin_y in coin_locations:
        dist = np.linalg.norm([coin_x - agent_location[0], coin_y - agent_location[1]])
        if dist < closest_dist:
            closest_dist = dist
            closest_coin = [coin_x, coin_y]

    # the next direction to be closer to the closest coin
    if closest_coin is not None:
        x, y = closest_coin
        if   x - agent_location[0] > 0: features[0] = 1   # RIGHT
        elif x - agent_location[0] < 0: features[1] = 1   # LEFT

        if   y - agent_location[1] > 0: features[2] = 1   # DOWN
        elif y - agent_location[1] < 0: features[3] = 1   # UP

    return [features[3], features[0], features[2], features[1], 0 , 0]

# avoid bomb


def bomb_crate(agent_location, game_state):
    [x, y] = agent_location
    field = game_state['field']

    features = np.zeros(len(ACTIONS))

    if 1 in field[x-4:x+4, y] or 1 in field[x, y-4:y+4]:
        features[5] = 1 # BOMB
    
    return features

def find_closest_crates(agent_location, game_state): 
    crate_locations = game_state['coins']

    features = np.zeros(len(ACTIONS))
    closest_crate = None
    closest_dist = 100

    # find the closest coin
    for crate_x, crate_y in crate_locations:
        dist = np.linalg.norm([crate_x - agent_location[0], crate_y - agent_location[1]])
        if dist < closest_dist:
            closest_dist = dist
            closest_crate = [crate_x, crate_y]

    # the next direction to be closer to the closest coin
    if closest_crate is not None:
        x, y = closest_crate
        if   x - agent_location[0] > 0: features[0] = 1   # RIGHT
        elif x - agent_location[0] < 0: features[1] = 1   # LEFT

        if   y - agent_location[1] > 0: features[2] = 1   # DOWN
        elif y - agent_location[1] < 0: features[3] = 1   # UP

    return [features[3], features[0], features[2], features[1], 0 , 0]

def get_crate_location(game_state):
    crate_location = []

    field = game_state['field']
    for x in range(len(field)):
        for y in range(len(field[0])):
            if field[x][y] == 1:
                crate_location.append([x, y])

    return crate_location
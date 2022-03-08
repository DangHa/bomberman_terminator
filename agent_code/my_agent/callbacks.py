import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAITED', 'BOMB']

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
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
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
    # print("Transitions: ", self.transitions)
    # print("Weights: ", self.weights)
    # calculate the Q with state_to_features from self.transition and self.weights

    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


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

    # Features from s
    arena = game_state['field']
    _, score, bombs_left, (agent_x, agent_y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5

    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    coin_direction = find_coins([agent_x, agent_y], coins)

    # the features can be used are score, bombs_left, bomb_maps, coins

    features = [score, bombs_left, bomb_map, coin_direction] 

    features = [0, 0, 0, 0]
    return features

def find_coins(agent_location, coin_locations):
    coin_direction = np.zeros(6)
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
        if   x - agent_location[0] > 0: coin_direction[0] = 1   # DOWN
        elif x - agent_location[0] < 0: coin_direction[1] = 1   # UP

        if   y - agent_location[1] > 0: coin_direction[2] = 1   # RIGHT
        elif y - agent_location[1] < 0: coin_direction[3] = 1   # LEFT

    return coin_direction

def find_crates(agent_location, crate_locations):
    ...

def bomb_crates(agent_location, crate_locations):
    ...

def bomb_opponents(agent_location, opponet_locations):
    ...

def avoid_bombs(agent_location, bomb_map):
    ...
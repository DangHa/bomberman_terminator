import os
import pickle
import random

from collections import deque

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = 9  #Add feature names
ACTION_HISTORY_SIZE = 4

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
    

    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.init_num = 1
        self.min_split = 100
        self.former_action = deque(maxlen=ACTION_HISTORY_SIZE)

        temp = []
        for i in range(len(ACTIONS)):
            tree = RandomForestRegressor(n_estimators=self.init_num, min_samples_split=self.min_split, warm_start=True)
            temp.append(tree)
        
        self.forests = np.array(temp)
        for i in range(len(ACTIONS)):
            random_s = np.random.rand(10, FEATURES)
            random_Y = np.random.rand(10)
            self.forests[i].fit(random_s, random_Y)
        self.logger.info("Forests are initialized.")

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.forests = pickle.load(file)
            self.former_action = deque(maxlen=ACTION_HISTORY_SIZE)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.epsilon = 0.1
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Training Mode (Exploration): Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0]) #PHASE 1 (No crates yet)
    else:

        #logger comment
        self.logger.info(f"Last 4 stored actions:  {self.former_action}")
        if len(self.former_action) == 4:
            if (self.former_action[0] == self.former_action[2]) and (self.former_action[1] == self.former_action[3]) and (self.former_action[0] != self.former_action[1]) and (self.former_action[2] != self.former_action[3]):
                #up-down-up-down
                if self.former_action[0] in ['UP', 'DOWN']:
                    if self.former_action[1] in ['UP', 'DOWN']:
                        self.logger.info(f"Your are in a loop!")

                #left-right-left-right
                if self.former_action[0] in ['LEFT', 'RIGHT']:
                    if self.former_action[1] in ['LEFT', 'RIGHT']:
                        self.logger.info(f"Your are in a loop!")

        features_for_all_actions = feature_for_all_action(game_state, self)

        #get the best action
        self.logger.debug("Training Mode (Exploitation) / Playing: Choosing action based Tree.")

        Y = np.array([self.forests[i].predict(features_for_all_actions)[0] for i in range(len(ACTIONS))])
        action_index = np.argmax(Y)
        action = ACTIONS[action_index]

        print("Action: ", action)

        #to store self.action_loop_result_before_taken_action
        state_to_features(game_state, action, self)
        self.logger.info(f"Value of self.action_loop_result_before_taken_action: {self.action_loop_result_before_taken_action}")
        
        
        self.former_action.append(action)
        self.logger.info(f'CHOOSEN ACTION: {action}')

        return action

def feature_for_all_action(game_state, self):
    features_for_action_1 = state_to_features(game_state, ACTIONS[0], self)
    features_for_action_2 = state_to_features(game_state, ACTIONS[1], self)
    features_for_action_3 = state_to_features(game_state, ACTIONS[2], self)
    features_for_action_4 = state_to_features(game_state, ACTIONS[3], self)
    features_for_action_5 = state_to_features(game_state, ACTIONS[4], self)
    features_for_action_6 = state_to_features(game_state, ACTIONS[5], self)

    self.logger.info(f"Value for Action UP: {state_to_features(game_state, ACTIONS[0], self)[2]}")
    self.logger.info(f"Value for Action RIGHT: {state_to_features(game_state, ACTIONS[1], self)[2]}")
    self.logger.info(f"Value for Action DOWN: {state_to_features(game_state, ACTIONS[2], self)[2]}")
    self.logger.info(f"Value for Action LEFT: {state_to_features(game_state, ACTIONS[3], self)[2]}")

    features_for_all_actions = np.array([
        features_for_action_1,
        features_for_action_2,
        features_for_action_3,
        features_for_action_4,
        features_for_action_5,
        features_for_action_6
    ])

    return features_for_all_actions

def state_to_features(game_state: dict, action, self) -> np.array:

    if game_state is None:
        return None

    a = f0()
    b = runs_into_wall_crate(game_state, action, self)
    c = action_loop(game_state, action, self)
    d = runs_towards_closest_coin_but_not_wall_or_crate(game_state, action, self)
    e = runs_away_from_closest_coin_but_not_wall_or_crate(game_state, action, self)
    f = drop_bomb_if_in_range_of_crate(  find_closest_crates(game_state, action, self)  , game_state, action, self)
    g = runs_towards_closest_crate_but_not_wall_or_crate(  find_closest_crates(game_state, action, self)  , game_state, action, self)

    #only for coin heaven stage
    at_end1 = bomb_dropped(game_state, action, self)
    at_end2 = waited(game_state, action, self)

    return np.array([a, b, c, d, e, f, g, at_end1, at_end2])


def f0():
    return 1


def runs_into_wall_crate(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #CAUTION x and y are switched in 'field' variable
    moved_up = game_state['field'][agent_coord_y-1][agent_coord_x] #Up
    moved_ri = game_state['field'][agent_coord_y][agent_coord_x+1] #Right
    moved_do = game_state['field'][agent_coord_y+1][agent_coord_x] #Down
    moved_le = game_state['field'][agent_coord_y][agent_coord_x-1] #Left

    # self.logger.info(f"Field: {game_state['field']}") 
    # self.logger.info(f"Value of upper tile: {game_state['field'][agent_coord_y-1][agent_coord_x]}") 
    # self.logger.info(f"Coordinates: {agent_coord_x}, {agent_coord_y}") 
    

    if action == 'UP' and moved_up != 0:
        return 1
    if action == 'RIGHT' and moved_ri != 0:
        return 1
    if action == 'DOWN' and moved_do != 0:
        return 1  
    if action == 'LEFT' and moved_le != 0:
        return 1

    return 0


def action_loop(game_state, action, self):

    #if in action loop
    if len(self.former_action) == 4:
        if (self.former_action[0] == self.former_action[2]) and (self.former_action[1] == self.former_action[3]) and (self.former_action[0] != self.former_action[1]) and (self.former_action[2] != self.former_action[3]):
            #up-down-up-down
            if self.former_action[0] in ['UP', 'DOWN']:
                if self.former_action[1] in ['UP', 'DOWN']:
                    if action == self.former_action[0]:
                        self.action_loop_result_before_taken_action = 1
                        return -1
                    #in case you want to disable 'up-down-up-wait' loop
                    # if action == 'WAIT':
                    #     self.action_loop_result_before_taken_action = 1
                    #     return -1                   

            #left-right-left-right
            if self.former_action[0] in ['LEFT', 'RIGHT']:
                if self.former_action[1] in ['LEFT', 'RIGHT']:
                    if action == self.former_action[0]:
                        self.action_loop_result_before_taken_action = 1
                        return -1
                    # if action == 'WAIT':
                    #     self.action_loop_result_before_taken_action = 1
                    #     return -1   

    self.action_loop_result_before_taken_action = 0
    return 0


def runs_towards_closest_coin_but_not_wall_or_crate(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    coin_locations = game_state['coins']
    closest_coin = None
    closest_dist = 100

    # find the closest coin
    for coin_x, coin_y in coin_locations:
        dist = np.linalg.norm([coin_x - agent_coord_x, coin_y - agent_coord_y])
        if dist < closest_dist:
            closest_dist = dist
            closest_coin = [coin_x, coin_y]

    # the next direction to be closer to the closest coin
    if closest_coin is not None:
        
        x, y = closest_coin

        if action == 'UP':
            #check if distance along changing axis is reduced and there is no wall or crate on the field u go to
            if abs(y - (agent_coord_y - 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #to prevent up-down-loop: checks if movement would bring agent into:  _|_|nearest-coin|_|_|_|agent|_|_
                if ((agent_coord_y - 1) % 2) != 0: 
                    return 1
                else:
                    if abs(y - (agent_coord_y - 1)) != 0 :
                        return 1
                    #if so, only give positive none zero feedback if agent would get the coin
                    else:
                        if abs(x - agent_coord_x) == 0:
                            return 1

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                if ((agent_coord_x + 1) % 2) != 0: 
                    return 1
                else:
                    if abs(x - (agent_coord_x + 1)) != 0 :
                        return 1
                    #if so, only give positive none zero feedback if agent would get the coin
                    else:
                        if abs(y - agent_coord_y) == 0:
                            return 1

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                if ((agent_coord_y + 1) % 2) != 0: 
                    return 1
                else:
                    if abs(y - (agent_coord_y + 1)) != 0 :
                        return 1
                    #if so, only give positive none zero feedback if agent would get the coin
                    else:
                        if abs(x - agent_coord_x) == 0:
                            return 1

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                if ((agent_coord_x - 1) % 2) != 0: 
                    return 1
                else:
                    if abs(x - (agent_coord_x - 1)) != 0 :
                        return 1
                    #if so, only give positive none zero feedback if agent would get the coin
                    else:
                        if abs(y - agent_coord_y) == 0:
                            return 1


    return 0


def runs_away_from_closest_coin_but_not_wall_or_crate(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    coin_locations = game_state['coins']
    closest_coin = None
    closest_dist = 100

    # find the closest coin
    for coin_x, coin_y in coin_locations:
        dist = np.linalg.norm([coin_x - agent_coord_x, coin_y - agent_coord_y])
        if dist < closest_dist:
            closest_dist = dist
            closest_coin = [coin_x, coin_y]

    # the next direction to be closer to the closest coin
    if closest_coin is not None:
        
        x, y = closest_coin

        if action == 'UP':
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                return 1

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                return 1

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                return 1

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                return 1


    return 0


# helper function
def find_closest_crates(game_state, action, self): 
    
    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #generate a 'crate_locations' list like the coin_locations
    x_vals = np.where(game_state['field'] == 1)[1]
    y_vals = np.where(game_state['field'] == 1)[0]
    crate_locations = [(x, y) for (x, y) in zip(x_vals, y_vals) if y is not None]


    closest_crate = None
    closest_dist = 100

    # find the closest crate
    for crate_x, crate_y in crate_locations:
        dist = np.linalg.norm([crate_x - agent_coord_x, crate_y - agent_coord_y])
        if dist < closest_dist:
            closest_dist = dist
            closest_crate = [crate_x, crate_y]
    
    return closest_crate


def drop_bomb_if_in_range_of_crate(closest_crate, game_state, action, self): # closest_crate = return of 'find_closest_crates(game_state, action, self)'

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    # the next direction to be closer to the closest crate
    if closest_crate is not None:
        
        x, y = closest_crate

        #if in range give points for dropping bomb if possible and 0 else

        #possible to drop a bomb
        if game_state['self'][2]:
            #range bomb in y range of explosion
            if abs(y - agent_coord_y) <= 3:
                if abs(x - agent_coord_x) == 0:
                    #no wall in between
                    if (agent_coord_x % 2) != 0 :
                        if action == 'BOMB':
                            return 1

            #range bomb in x range of explosion
            if abs(x - agent_coord_x) <= 3:
                if abs(y - agent_coord_y) == 0:
                    #no wall in between
                    if (agent_coord_y % 2) != 0 :
                        if action == 'BOMB':
                            return 1

    return 0


def runs_towards_closest_crate_but_not_wall_or_crate(closest_crate, game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    # the next direction to be closer to the closest crate
    if closest_crate is not None:
        
        x, y = closest_crate


        if action == 'UP':
            #check if distance along changing axis is reduced and there is no wall or crate on the field u go to
            if abs(y - (agent_coord_y - 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #if agent would move into crate dont give reward
                if abs(y - (agent_coord_y - 1)) == 0 and abs(x - agent_coord_x) == 0:
                    return 0
                #to prevent up-down-loop: checks if movement would bring agent into:  _|_|nearest-coin|_|_|_|agent|_|_
                if ((agent_coord_y - 1) % 2) != 0: 
                    return 1
                else:
                    if abs(y - (agent_coord_y - 1)) != 0 :
                        return 1


        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                if abs(x - (agent_coord_x + 1)) == 0 and abs(y - agent_coord_y) == 0:
                    return 0
                if ((agent_coord_x + 1) % 2) != 0: 
                    return 1
                else:
                    if abs(x - (agent_coord_x + 1)) != 0 :
                        return 1
      

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                if abs(y - (ag5ent_coord_y + 1)) == 0 and abs(x - agent_coord_x) == 0:
                    return 0                
                if ((agent_coord_y + 1) % 2) != 0: 
                    return 1
                else:
                    if abs(y - (agent_coord_y + 1)) != 0 :
                        return 1
              

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                if abs(x - (agent_coord_x - 1)) == 0 and abs(y - agent_coord_y) == 0:
                    return 0                
                if ((agent_coord_x - 1) % 2) != 0: 
                    return 1
                else:
                    if abs(x - (agent_coord_x - 1)) != 0 :
                        return 1


    return 0


#function only needed to test coin-heaven level
def bomb_dropped(game_state, action, self):

    if action == 'BOMB':
        return 1

    return 0


#function only needed to test coin-heaven level
def waited(game_state, action, self):

    if action == 'WAIT':
        return 1

    return 0

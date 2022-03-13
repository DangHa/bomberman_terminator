import os
import pickle
import random

import numpy as np
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = 6
ACTION_HISTORY_SIZE = 4

#done
def setup(self):

    #if no model exists
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("SETTING UP MODEL FROM SCRATCH")
        weights = np.random.rand(FEATURES)
        self.former_action = deque(maxlen=ACTION_HISTORY_SIZE)
        self.action_loop_result_before_taken_action = 0
        self.a = 0 #for action loop
        self.model = weights
    else:
        self.former_action = deque(maxlen=ACTION_HISTORY_SIZE)
        self.action_loop_result_before_taken_action = 0
        self.a = 0 #for action loop
        self.logger.info("LOADING MODEL FROM FILE")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


#done
def act(self, game_state: dict) -> str:

    #Explore
    if self.train and random.random() < self.epsilon:
        action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0]) #PHASE 1 (No crates yet)
        self.former_action.append(action)
        self.logger.info(f'RANDOM ACTION: {action}')
        return action
    
    #Exploit (choose best action)
    else:

        self.logger.info(f"Last 4 stored actions:  {self.former_action}")
        if len(self.former_action) == 4:
            if (self.former_action[0] == self.former_action[2]) and (self.former_action[1] == self.former_action[3]) and (self.former_action[0] != self.former_action[1]) and (self.former_action[2] != self.former_action[3]):
                self.logger.info(f"Your are in a loop!")


        weights = self.model
        features_for_action_1 = state_to_features(game_state, ACTIONS[0], self)
        features_for_action_2 = state_to_features(game_state, ACTIONS[1], self)
        features_for_action_3 = state_to_features(game_state, ACTIONS[2], self)
        features_for_action_4 = state_to_features(game_state, ACTIONS[3], self)
        features_for_action_5 = state_to_features(game_state, ACTIONS[4], self)
        features_for_action_6 = state_to_features(game_state, ACTIONS[5], self)

        self.logger.info(f"Value of for Action UP: {state_to_features(game_state, ACTIONS[0], self)[4]}")
        self.logger.info(f"Value of for Action RIGHT: {state_to_features(game_state, ACTIONS[1], self)[4]}")
        self.logger.info(f"Value of for Action DOWN: {state_to_features(game_state, ACTIONS[2], self)[4]}")
        self.logger.info(f"Value of for Action LEFT: {state_to_features(game_state, ACTIONS[3], self)[4]}")

        features_for_all_actions = np.array([
            features_for_action_1,
            features_for_action_2,
            features_for_action_3,
            features_for_action_4,
            features_for_action_5,
            features_for_action_6
        ])

        index_of_best_action = ((weights * features_for_all_actions).sum(axis=1)).argmax(axis=0)
        action = ACTIONS[index_of_best_action]

        #to store self.action_loop_result_before_taken_action
        state_to_features(game_state, action, self)
        self.logger.info(f"Value of self.action_loop_result_before_taken_action: {self.action_loop_result_before_taken_action}")
        
        d = 0
        self.former_action.append(action)
        self.logger.info(f'CHOOSEN ACTION: {action}')

        return action




def state_to_features(game_state: dict, action, self) -> np.array:

    if game_state is None:
        return None

    a = f0()
    b = runs_into_wall_crate(game_state, action, self)
    c = bomb_dropped(game_state, action, self)
    d = waited(game_state, action, self)
    e = action_loop(game_state, action, self)
    f = runs_towards_closest_coin_but_not_wall_or_crate(game_state, action, self)


    return np.array([a, b, c, d, e, f])


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


def bomb_dropped(game_state, action, self):

    if action == 'BOMB':
        return 1

    return 0


def waited(game_state, action, self):

    if action == 'WAIT':
        return 1

    return 0


def action_loop(game_state, action, self):

    #if in action
    if len(self.former_action) == 4:
        if (self.former_action[0] == self.former_action[2]) and (self.former_action[1] == self.former_action[3]) and (self.former_action[0] != self.former_action[1]) and (self.former_action[2] != self.former_action[3]):
            if action == self.former_action[0]:
                self.action_loop_result_before_taken_action = 1
                return 1

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
            if abs(y - (agent_coord_y - 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                return 1

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                return 1

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                return 1

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                return 1


    return 0






# def find_coins(game_state):
#     agent_coord_x = game_state['self'][3][0]
#     agent_coord_y = game_state['self'][3][1]
#     agent_location = [agent_coord_x, agent_coord_y]
#     coin_locations = game_state['coins']

#     features = np.zeros(len(ACTIONS))
#     closest_coin = None
#     closest_dist = 100 #careful!

#     # find the closest coin
#     for coin_x, coin_y in coin_locations:
#         dist = np.linalg.norm([coin_x - agent_location[0], coin_y - agent_location[1]])
#         if dist < closest_dist:
#             closest_dist = dist
#             closest_coin = [coin_x, coin_y]


#     if closest_coin is not None:
#         x, y = closest_coin

#         dist_x = abs(agent_location[0] - x)
#         dist_y = abs(agent_location[1] - y)

#         return np.array([dist_x, dist_y])


#         if dist_x == 0:
#             dist_y_inverse = 1/dist_y
#             return np.array([2, dist_y_inverse])

#         if dist_y == 0:
#             dist_x_inverse = 1/dist_x
#             return np.array([dist_x_inverse, 2])       
        
#         dist_x_inverse = 1/dist_x
#         dist_y_inverse = 1/dist_y

#         return np.array([dist_x_inverse, dist_y_inverse])

#     else:
#         return np.array([0, 0])
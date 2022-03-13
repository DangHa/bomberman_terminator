import os
import pickle
import random

import numpy as np
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = 7
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


        weights = self.model
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

        index_of_best_action = ((weights * features_for_all_actions).sum(axis=1)).argmax(axis=0)
        action = ACTIONS[index_of_best_action]


        #to store self.action_loop_result_before_taken_action
        state_to_features(game_state, action, self)
        self.logger.info(f"Value of self.action_loop_result_before_taken_action: {self.action_loop_result_before_taken_action}")
        
        
        self.former_action.append(action)
        self.logger.info(f'CHOOSEN ACTION: {action}')

        return action




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

    # #only for coin heaven stage
    # at_end1 = bomb_dropped(game_state, action, self)
    # at_end2 = waited(game_state, action, self)

    return np.array([a, b, c, d, e, f, g])


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
                if abs(y - (agent_coord_y + 1)) == 0 and abs(x - agent_coord_x) == 0:
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







# #function only needed to test coin-heaven level
# def bomb_dropped(game_state, action, self):

#     if action == 'BOMB':
#         return 1

#     return 0


# #function only needed to test coin-heaven level
# def waited(game_state, action, self):

#     if action == 'WAIT':
#         return 1

#     return 0
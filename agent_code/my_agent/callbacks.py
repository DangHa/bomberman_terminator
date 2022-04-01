import os
import pickle
import random

import numpy as np
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT'] #action space reduced for first stage
FEATURES = 13

#done
def setup(self):

    #if no model exists
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("SETTING UP MODEL FROM SCRATCH")
        q_table = np.zeros((FEATURES,len(ACTIONS)))
        self.model = q_table

    #if exists
    else:
        self.logger.info("LOADING MODEL FROM FILE")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    #hyper params
    self.epsilon = 0.10

    #useful tracking
    self.random_or_choosen = 0 #only for logger


#done
def act(self, game_state: dict) -> str:

    #bug fix, to actually update model when using '--n-rounds' flag command
    if game_state["step"] == 1:
        #if no model exists
        if not os.path.isfile("my-saved-model.pt"):
            self.logger.info("SETTING UP MODEL FROM SCRATCH")
            q_table = np.zeros((FEATURES,len(ACTIONS)))
            self.model = q_table

        #if exists
        else:
            self.logger.info("LOADING MODEL FROM FILE")
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)


    #for logger
    self.logger.info(f'\n------------------------------------ Step: {game_state["step"]}')
    


    #Explore
    if self.train and random.random() < self.epsilon:
        action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25]) #action space reduced for first stage      
        
        self.random_or_choosen = 1
    
    #Exploit (choose best action)
    else:

        q_table = self.model
        
        state_index = get_state_index_from_game_state(game_state, self)
        action = np.argmax(q_table[state_index])
       
        
        self.random_or_choosen = 2

       
    #for logger
    if self.random_or_choosen == 2:
        self.logger.info(f"CHOSEN ACTION: {action}")
    elif self.random_or_choosen == 1:
        self.logger.info(f"RANDOM ACTION: {action}")



    return action




def get_state_index_from_game_state(game_state, self):

    if game_state is None:
        return None

    features_for_action_1 = runs_into_wall_crate(game_state, ACTIONS[0], self)
    features_for_action_2 = runs_into_wall_crate(game_state, ACTIONS[1], self)
    features_for_action_3 = runs_into_wall_crate(game_state, ACTIONS[2], self)
    features_for_action_4 = runs_into_wall_crate(game_state, ACTIONS[3], self)
    goes_towards_coin = closest_coin_but_not_wall_or_crate(game_state, self)


    features_for_all_actions = np.array([
        features_for_action_1,
        features_for_action_2,
        features_for_action_3,
        features_for_action_4,
        goes_towards_coin
    ])

    state_index = feature_to_index(features_for_all_actions)

    return state_index


def feature_to_index(features):

    index = 0

    if features[0] == 1: 
        index +=1
    if features[1] == 1: 
        index +=2
    if features[2] == 1: 
        index +=4
    if features[3] == 1: 
        index +=8

    if features[4] == 1: 
        index += 2 * 16
    if features[4] == 2: 
        index += 2 * 16
    if features[4] == 3: 
        index += 4 * 16
    if features[4] == 4: 
        index += 5 * 16

    return index





#done
def runs_into_wall_crate(game_state, action, self):

    if game_state is None:
        return None

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #CAUTION x and y are have to be accessed as follows in the 'field' variable
    moved_up = game_state['field'][agent_coord_x][agent_coord_y-1] #Up    (definitly correct)
    moved_ri = game_state['field'][agent_coord_x+1][agent_coord_y] #Right (definitly correct)
    moved_do = game_state['field'][agent_coord_x][agent_coord_y+1] #Down  (definitly correct)
    moved_le = game_state['field'][agent_coord_x-1][agent_coord_y] #Left  (definitly correct)


    #if no crates as obsticles, this is sufficient
    # if action in ['UP','DOWN'] and ((moved_up != 0) or (moved_do != 0)) :
    #     return 1
    # if action in ['RIGHT','LEFT'] and ((moved_ri != 0) or (moved_le != 0)) :
    #     return 1

    
    if action == 'UP' and moved_up != 0:
        return 1
    if action == 'RIGHT' and moved_ri != 0:
        return 1
    if action == 'DOWN' and moved_do != 0:
        return 1
    if action == 'LEFT' and moved_le != 0:
        return 1

    return 0




#done
def closest_coin_but_not_wall_or_crate(game_state, self):

    if game_state is None:
        return None

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


    self.logger.info(f"Coordinates of coins: \n {coin_locations}")
    self.logger.info(f"Coordinates of closest coin: {closest_coin}")

    a = 0

    # the next direction to be closer to the closest coin
    if closest_coin is not None:
        
        x, y = closest_coin

        #coin above agent
        #check if distance along changing axis is reduced and there is no wall or crate on the field u go to
        if abs(y - (agent_coord_y - 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
            #to prevent up-down-loop: checks if movement would bring agent into:  _|_|nearest-coin|_|_|_|agent|_|_
            if ((agent_coord_y - 1) % 2) != 0: 
                a = 1
            else:
                if abs(y - (agent_coord_y - 1)) != 0 :
                    a = 1
                #if so, only give positive none zero feedback if agent would get the coin
                else:
                    if abs(x - agent_coord_x) == 0:
                        a = 1

        #coin right from agent
        if abs(x - (agent_coord_x + 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
            if ((agent_coord_x + 1) % 2) != 0: 
                a = 2
            else:
                if abs(x - (agent_coord_x + 1)) != 0 :
                    a = 2
                #if so, only give positive none zero feedback if agent would get the coin
                else:
                    if abs(y - agent_coord_y) == 0:
                        a = 2

        #coin below agent
        if abs(y - (agent_coord_y + 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
            if ((agent_coord_y + 1) % 2) != 0: 
                a = 3
            else:
                if abs(y - (agent_coord_y + 1)) != 0 :
                    a = 3
                #if so, only give positive none zero feedback if agent would get the coin
                else:
                    if abs(x - agent_coord_x) == 0:
                        a = 3

        #coin left from agent
        if abs(x - (agent_coord_x - 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
            if ((agent_coord_x - 1) % 2) != 0: 
                a = 4
            else:
                if abs(x - (agent_coord_x - 1)) != 0 :
                    a = 4
                #if so, only give positive none zero feedback if agent would get the coin
                else:
                    if abs(y - agent_coord_y) == 0:
                        a = 4


    return a
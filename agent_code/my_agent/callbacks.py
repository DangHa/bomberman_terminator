import os
import pickle
import random

import numpy as np
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = 16
ACTION_HISTORY_SIZE = 4
STATE_HISTORY_SIZE = 2

#done
def setup(self):

    #if no model exists
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("SETTING UP MODEL FROM SCRATCH")
        weights = np.random.rand(FEATURES)
        self.former_action = deque(maxlen=ACTION_HISTORY_SIZE)
        self.former_state = deque(maxlen=STATE_HISTORY_SIZE)
        self.action_loop_result_before_taken_action = 0
        self.a = 0 #for action loop
        self.model = weights
    else:
        self.former_action = deque(maxlen=ACTION_HISTORY_SIZE)
        self.former_state = deque(maxlen=STATE_HISTORY_SIZE)
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

        # self.logger.info(f"Value for Action UP: {state_to_features(game_state, ACTIONS[0], self)[2]}")
        # self.logger.info(f"Value for Action RIGHT: {state_to_features(game_state, ACTIONS[1], self)[2]}")
        # self.logger.info(f"Value for Action DOWN: {state_to_features(game_state, ACTIONS[2], self)[2]}")
        # self.logger.info(f"Value for Action LEFT: {state_to_features(game_state, ACTIONS[3], self)[2]}")

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
        # self.logger.info(f"Value of self.action_loop_result_before_taken_action: {self.action_loop_result_before_taken_action}")
        
        
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
    h = runs_away_from_closest_crate_but_not_wall_or_crate(  find_closest_crates(game_state, action, self)  , game_state, action, self)
    i = get_away_from_bomb(   find_planted_bombs_in_dangerous_range(game_state, action, self)   ,   find_closest_crates(game_state, action, self),   game_state, action, self)
    j = get_out_of_bomb_range(   find_planted_bombs_in_dangerous_range(game_state, action, self)   ,   find_closest_crates(game_state, action, self),   game_state, action, self)
    k = goes_towards_crate_trap(   find_planted_bombs_in_dangerous_range(game_state, action, self)   ,   find_closest_crates(game_state, action, self),   game_state, action, self)
    l = bomb_dropped_although_not_possible(game_state, action, self)
    m = runs_into_explosion(game_state, action, self)

    n = runs_into_bomb_range_without_dying(game_state, action, self)
    o = runs_into_bomb_range_with_dying(game_state, action, self)

    at_end2 = waited(game_state, action, self)


    

    # #only for coin heaven stage
    # at_end1 = bomb_dropped(game_state, action, self)
    # at_end2 = waited(game_state, action, self)


    return np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, at_end2])


def f0():
    return 1


def runs_into_wall_crate(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #CAUTION x and y are switched in 'field' variable
    moved_up = game_state['field'][agent_coord_x][agent_coord_y-1] #Up
    moved_ri = game_state['field'][agent_coord_x+1][agent_coord_y] #Right (definitly correct)
    moved_do = game_state['field'][agent_coord_x][agent_coord_y+1] #Down
    moved_le = game_state['field'][agent_coord_x-1][agent_coord_y] #Left  (definitly correct)

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


#ATENTION!!!! I DID SOME WEIRD ASS SHIT IN THE FOLLOWING FUNCS: 
# At the begining of coding I switched x and y for the game_state['field'] variable, so 
# the arrays of coin and crate coordinates are switched I think (because they work now even though 
# I am using switched x and y for game_state['field']) ... sorry for that but I wont change all my 
# code now if it works just fine ... though a bit harder for you to understand it maybe


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

    # the next direction to be farther from the closest coin
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

    #generate a 'crate_locations' list like the coin_locations, so like:  [(x,y), (x,y), (x,y)]
    x_vals = np.where(game_state['field'] == 1)[1]
    y_vals = np.where(game_state['field'] == 1)[0]
    crate_locations = [(x, y) for (x, y) in zip(y_vals, x_vals) if y is not None]


    closest_crate = None
    closest_dist = 100

    # find the closest crate
    for crate_x, crate_y in crate_locations:
        dist = np.linalg.norm([crate_x - agent_coord_x, crate_y - agent_coord_y])
        if dist < closest_dist:
            closest_dist = dist
            closest_crate = [crate_x, crate_y]
    
    # self.logger.info(f"Coordinates of crates: \n {crate_locations}")
    # self.logger.info(f"Coordinates of closest crates: \n {closest_crate}")

    return closest_crate


def drop_bomb_if_in_range_of_crate(closest_crate, game_state, action, self): # closest_crate = return of 'find_closest_crates(game_state, action, self)'

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    # the next direction to be closer to the closest crate
    if closest_crate is not None:
        
        x, y = closest_crate

        #if in range give points for dropping bomb if possible and 0 else

        if action == 'BOMB':
            #possible to drop a bomb
            if game_state['self'][2]:
                #range bomb in y range of explosion
                if abs(y - agent_coord_y) <= 3 and abs(x - agent_coord_x) == 0:
                    #no wall in between
                    if (agent_coord_x % 2) != 0 :
                        return 1

                #range bomb in x range of explosion
                if abs(x - agent_coord_x) <= 3 and abs(y - agent_coord_y) == 0:
                    #no wall in between
                    if (agent_coord_y % 2) != 0 :
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


def runs_away_from_closest_crate_but_not_wall_or_crate(closest_crate, game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    # the next direction to be closer to the closest crate
    if closest_crate is not None:
        
        x, y = closest_crate


        if action == 'UP':
            #check if distance along changing axis is reduced and there is no wall or crate on the field u go to
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #if agent would move into crate dont give reward
                if abs(y - (agent_coord_y - 1)) == 0 and abs(x - agent_coord_x) == 0:
                    return 0
                else:
                    return 1


        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                if abs(x - (agent_coord_x + 1)) == 0 and abs(y - agent_coord_y) == 0:
                    return 0
                else:
                    return 1
      

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                if abs(y - (agent_coord_y + 1)) == 0 and abs(x - agent_coord_x) == 0:
                    return 0                
                else:
                    return 1
              

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                if abs(x - (agent_coord_x - 1)) == 0 and abs(y - agent_coord_y) == 0:
                    return 0                
                else:
                    return 1


    return 0


#helper helper function
def is_bomb_dangerous(bomb, agent_coord_x, agent_coord_y):

    if bomb is not None:

        #agent in y range of explosion
        if abs(bomb[1] - agent_coord_y) <= 3 and abs(bomb[0] - agent_coord_x) == 0:
            #no wall in between
            if (agent_coord_x % 2) != 0 :
                return True
        
        #range bomb in x range of explosion
        if abs(bomb[0] - agent_coord_x) <= 3 and abs(bomb[1] - agent_coord_y) == 0:
            #no wall in between
            if (agent_coord_y % 2) != 0 :
                return True

    return False


#helper function
def find_planted_bombs_in_dangerous_range(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #generate a 'bomb_locations' list like the coin_locations, so like:  [(x,y), (x,y), (x,y)]
    bomb_locations = [item[0] for item in game_state['bombs']]
    
    bomb_infos = game_state['bombs']

    closest_bomb1 = (None, 0)
    closest_bomb2 = (None, 0)
    closest_bomb3 = (None, 0)
    closest_dist = 100

    # find the 3 closest bombs
    if len(bomb_locations) != 0:
        #sort 'bomb_locations' list according to cartesian distance to agent
        bomb_locations_sorted = sorted( bomb_locations, key = lambda coord: np.linalg.norm([coord[0] - agent_coord_x, coord[1] - agent_coord_y] ) )

        bomb_infos_sorted = sorted( bomb_infos, key = lambda coord: np.linalg.norm([coord[0][0] - agent_coord_x, coord[0][1] - agent_coord_y] ) )

        if len(bomb_infos) == 1:
            closest_bomb1 = bomb_infos_sorted[0]
        
        if len(bomb_infos) == 2:
            closest_bomb1 = bomb_infos_sorted[0]
            closest_bomb2 = bomb_infos_sorted[1]

        if len(bomb_infos) > 2:
            closest_bomb1 = bomb_infos_sorted[0]
            closest_bomb2 = bomb_infos_sorted[1]
            closest_bomb3 = bomb_infos_sorted[2]

    closest_bombs = [closest_bomb1, closest_bomb2, closest_bomb3]


    #check for which of these bombs the agent is in explosion range
    number_of_dangerous_bombs = 0

    if closest_bombs[0][0] is not None:
        if is_bomb_dangerous(closest_bombs[0][0], agent_coord_x, agent_coord_y):
            number_of_dangerous_bombs += 1
        else:
            closest_bomb1 = (None, 0)

    if closest_bombs[1][0] is not None:
        if is_bomb_dangerous(closest_bombs[1][0], agent_coord_x, agent_coord_y):
            number_of_dangerous_bombs += 1
        else:
            closest_bomb2 = (None, 0)
    
    if closest_bombs[2][0] is not None:
        if is_bomb_dangerous(closest_bombs[2][0], agent_coord_x, agent_coord_y):
            number_of_dangerous_bombs += 1
        else:
            closest_bomb3 = (None, 0)

    #define with new 'None' entries
    dangerous_bombs = [closest_bomb1, closest_bomb2, closest_bomb3]


    #only using the 2 closest bombs (delete if you want to implement 3 bomb behavior)
    dangerous_bombs = [closest_bomb1, closest_bomb2]
    if number_of_dangerous_bombs == 3:
        number_of_dangerous_bombs = 2

    return [dangerous_bombs, number_of_dangerous_bombs]


#here we would still need to solve crate trap problem for 2 crates (only solved for one)
# arg: 'dangerous_bombs, number_of_dangerous_bombs'  is  'find_planted_bombs_in_dangerous_range(game_state, action, self)'
# arg: 'closest_crate' is 'find_closest_crates(game_state, action, self)'
def get_away_from_bomb(dangerous_bombs_and_number_of_dangerous_bombs, closest_crate, game_state, action, self):
    
    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]


    if closest_crate is not None:
        crate_x, crate_y = closest_crate
        crate_in_expl_range = False

        #check if crate in explosion range of bomb
        #range bomb in y range of explosion
        if abs(crate_y - agent_coord_y) <= 3 and abs(crate_x - agent_coord_x) == 0:
            #no wall in between
            if (agent_coord_x % 2) != 0 :
                crate_in_expl_range = True
        #range bomb in x range of explosion
        if abs(crate_x - agent_coord_x) <= 3 and abs(crate_y - agent_coord_y) == 0:
            #no wall in between
            if (agent_coord_y % 2) != 0 :
                crate_in_expl_range = True

    crate_in_expl_range = False


    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]

        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y


        # the next direction to be farther from the bomb
        if action == 'UP':
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #avoid crate trap
                """
                ____W_
                _B__xC  (moved right in this case)
                ____W_
                """
                if crate_in_expl_range:
                    if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                        return 0
                return 1

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                        return 0
                return 1

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                        return 0
                return 1

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                        return 0
                return 1


    #case 2:  two bombs-------------------------------------------------------
    if number_of_dangerous_bombs == 2:

        #get the bombs which are not 'None'
        bomb_index = [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None]
        bomb_coord1 = dangerous_bombs[ bomb_index[0] ]
        bomb_coord2 = dangerous_bombs[ bomb_index[1] ]

        x1 = bomb_coord1[0]
        y1 = bomb_coord1[1]

        x2 = bomb_coord2[0]
        y2 = bomb_coord2[1]


        # the next direction to be farther from both bombs
        if action == 'UP':
            #1st: gets farther from at least ONE bomb,  2nd: doesnt get closer to other bomb
            #gets away from bomb 1
            if abs(y1 - (agent_coord_y - 1)) > abs(y1 - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y2 - (agent_coord_y - 1)) < abs(y2 - agent_coord_y)):
                    #dont run into crate trap 
                    """
                    ____W_
                    _B__xC (moved right in this case)
                    ____W_
                    """
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1
            #gets away from bomb 2
            if abs(y2 - (agent_coord_y - 1)) > abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y1 - (agent_coord_y - 1)) < abs(y1 - agent_coord_y)):
                    #dont run into crate trap 
                    """
                    ____W_
                    _B__xC (moved right in this case)
                    ____W_
                    """
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1


        if action == 'RIGHT':
            #gets away from bomb 1
            if abs(x1 - (agent_coord_x + 1)) > abs(x1 - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(x2 - (agent_coord_x + 1)) < abs(x2 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1
            #gets away from bomb 2
            if abs(x2 - (agent_coord_x + 1)) > abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                #doesnt get closer to bomb1
                if not (abs(x1 - (agent_coord_x + 1)) < abs(x1 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1


        if action == 'DOWN':
            #gets away from bomb 1
            if abs(y1 - (agent_coord_y + 1)) > abs(y1 - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y2 - (agent_coord_y + 1)) < abs(y2 - agent_coord_y)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1
            #gets away from bomb 2
            if abs(y2 - (agent_coord_y + 1)) > abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y1 - (agent_coord_y + 1)) < abs(y1 - agent_coord_y)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1


        if action == 'LEFT':
            #gets away from bomb 1
            if abs(x1 - (agent_coord_x - 1)) > abs(x1 - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(x2 - (agent_coord_x - 1)) < abs(x2 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                            return 0
                    return 1
            #gets away from bomb 2
            if abs(x2 - (agent_coord_x - 1)) > abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                #doesnt get closer to bomb1
                if not (abs(x1 - (agent_coord_x - 1)) < abs(x1 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                            return 0
                    return 1


        #if there is no way to move away from one bomb without getting closer to the other, we have the following situation:
        """
        W_W_W_W_W_W_W
        ___B__x__B___
        W_W_W_W_W_W_W

        """
        #sketched case
        if abs(y1 - agent_coord_y) == 0 and abs(y2 - agent_coord_y) == 0:
            if action == 'RIGHT':
                #only go right if you dont run directly on the field where a bomb is
                if abs(x1 - (agent_coord_x + 1)) != 0 and abs(x2 - (agent_coord_x + 1)) != 0 and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1
            if action == 'LEFT':
                if abs(x1 - (agent_coord_x - 1)) != 0 and abs(x2 - (agent_coord_x - 1)) != 0 and  ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                            return 0
                    return 1

        #90 degree rotated case
        if abs(x1 - agent_coord_x) == 0 and abs(x2 - agent_coord_x) == 0:
            if action == 'UP':
                if abs(y1 - (agent_coord_y - 1)) != 0 and abs(y2 - (agent_coord_y - 1)) != 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1
            if action == 'DOWN':
                if abs(y1 - (agent_coord_y + 1)) != 0 and abs(y2 - (agent_coord_y + 1)) != 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1



    #case 3:  three bombs (hell no, I aint gonna do this ... good luck)---------------------------------------



    return 0


#here we would still need to solve crate trap problem for 2 crates (only solved for one)
#exactly the same as 'get_away_from_bomb' only switch all return 0 and 1 values (except the final one)
def goes_towards_crate_trap(dangerous_bombs_and_number_of_dangerous_bombs, closest_crate, game_state, action, self):
    
    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    if closest_crate is not None:
        crate_x, crate_y = closest_crate
        crate_in_expl_range = False

        #check if crate in explosion range of bomb
        #range bomb in y range of explosion
        if abs(crate_y - agent_coord_y) <= 3 and abs(crate_x - agent_coord_x) == 0:
            #no wall in between
            if (agent_coord_x % 2) != 0 :
                crate_in_expl_range = True
        #range bomb in x range of explosion
        if abs(crate_x - agent_coord_x) <= 3 and abs(crate_y - agent_coord_y) == 0:
            #no wall in between
            if (agent_coord_y % 2) != 0 :
                crate_in_expl_range = True

    crate_in_expl_range = False

    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]

        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y


        # the next direction to be farther from the bomb
        if action == 'UP':
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #avoid crate trap
                """
                ____W_
                _B__xC  (moved right in this case)
                ____W_
                """
                if crate_in_expl_range:
                    if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                        return 1
                return 0

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                        return 1
                return 0

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                        return 1
                return 0

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                        return 1
                return 0


    #case 2:  two bombs-------------------------------------------------------
    if number_of_dangerous_bombs == 2:

        #get the bombs which are not 'None'
        bomb_index = [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None]
        bomb_coord1 = dangerous_bombs[ bomb_index[0] ]
        bomb_coord2 = dangerous_bombs[ bomb_index[1] ]

        x1 = bomb_coord1[0]
        y1 = bomb_coord1[1]

        x2 = bomb_coord2[0]
        y2 = bomb_coord2[1]


        # the next direction to be farther from both bombs
        if action == 'UP':
            #1st: gets farther from at least ONE bomb,  2nd: doesnt get closer to other bomb
            #gets away from bomb 1
            if abs(y1 - (agent_coord_y - 1)) > abs(y1 - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y2 - (agent_coord_y - 1)) < abs(y2 - agent_coord_y)):
                    #dont run into crate trap 
                    """
                    ____W_
                    _B__xC (moved right in this case)
                    ____W_
                    """
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0
            #gets away from bomb 2
            if abs(y2 - (agent_coord_y - 1)) > abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y1 - (agent_coord_y - 1)) < abs(y1 - agent_coord_y)):
                    #dont run into crate trap 
                    """
                    ____W_
                    _B__xC (moved right in this case)
                    ____W_
                    """
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0


        if action == 'RIGHT':
            #gets away from bomb 1
            if abs(x1 - (agent_coord_x + 1)) > abs(x1 - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(x2 - (agent_coord_x + 1)) < abs(x2 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0
            #gets away from bomb 2
            if abs(x2 - (agent_coord_x + 1)) > abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                #doesnt get closer to bomb1
                if not (abs(x1 - (agent_coord_x + 1)) < abs(x1 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0


        if action == 'DOWN':
            #gets away from bomb 1
            if abs(y1 - (agent_coord_y + 1)) > abs(y1 - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y2 - (agent_coord_y + 1)) < abs(y2 - agent_coord_y)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0
            #gets away from bomb 2
            if abs(y2 - (agent_coord_y + 1)) > abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y1 - (agent_coord_y + 1)) < abs(y1 - agent_coord_y)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0


        if action == 'LEFT':
            #gets away from bomb 1
            if abs(x1 - (agent_coord_x - 1)) > abs(x1 - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(x2 - (agent_coord_x - 1)) < abs(x2 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                            return 1
                    return 0
            #gets away from bomb 2
            if abs(x2 - (agent_coord_x - 1)) > abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                #doesnt get closer to bomb1
                if not (abs(x1 - (agent_coord_x - 1)) < abs(x1 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                            return 1
                    return 0


        #if there is no way to move away from one bomb without getting closer to the other, we have the following situation:
        """
        W_W_W_W_W_W_W
        ___B__x__B___
        W_W_W_W_W_W_W

        """
        #sketched case
        if abs(y1 - agent_coord_y) == 0 and abs(y2 - agent_coord_y) == 0:
            if action == 'RIGHT':
                #only go right if you dont run directly on the field where a bomb is
                if abs(x1 - (agent_coord_x + 1)) != 0 and abs(x2 - (agent_coord_x + 1)) != 0 and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0
            if action == 'LEFT':
                if abs(x1 - (agent_coord_x - 1)) != 0 and abs(x2 - (agent_coord_x - 1)) != 0 and  ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                            return 1
                    return 0

        #90 degree rotated case
        if abs(x1 - agent_coord_x) == 0 and abs(x2 - agent_coord_x) == 0:
            if action == 'UP':
                if abs(y1 - (agent_coord_y - 1)) != 0 and abs(y2 - (agent_coord_y - 1)) != 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0
            if action == 'DOWN':
                if abs(y1 - (agent_coord_y + 1)) != 0 and abs(y2 - (agent_coord_y + 1)) != 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                            return 1
                    return 0



    #case 3:  three bombs (hell no, I aint gonna do this ... good luck)---------------------------------------



    return 0


#here we would still need to solve crate trap problem for 2 crates (only solved for one)
# arg: 'dangerous_bombs, number_of_dangerous_bombs'  is  'find_planted_bombs_in_dangerous_range(game_state, action, self)'
# arg: 'closest_crate' is 'find_closest_crates(game_state, action, self)'
def get_out_of_bomb_range(dangerous_bombs_and_number_of_dangerous_bombs, closest_crate, game_state, action, self):
    
    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    if closest_crate is not None:
        crate_x, crate_y = closest_crate
        crate_in_expl_range = False

        #check if crate in explosion range of bomb
        #range bomb in y range of explosion
        if abs(crate_y - agent_coord_y) <= 3 and abs(crate_x - agent_coord_x) == 0:
            #no wall in between
            if (agent_coord_x % 2) != 0 :
                crate_in_expl_range = True
        #range bomb in x range of explosion
        if abs(crate_x - agent_coord_x) <= 3 and abs(crate_y - agent_coord_y) == 0:
            #no wall in between
            if (agent_coord_y % 2) != 0 :
                crate_in_expl_range = True
    else:
        crate_in_expl_range = False

    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]

        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y


        # the direction which would move agent out of explosion range (if there is no wall or crate)
        # (and not into crate trap (only one crate considered))
        if action == 'UP':
            if not ( is_bomb_dangerous(bomb_coord, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_coord, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #avoid crate trap
                """
                ____W_
                _B__xC  (moved right in this case)
                ____W_
                """
                if crate_in_expl_range:
                    if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                        return 0
                return 1

        if action == 'RIGHT':
            if not ( is_bomb_dangerous(bomb_coord, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_coord, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                        return 0
                return 1

        if action == 'DOWN':
            if not ( is_bomb_dangerous(bomb_coord, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_coord, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                        return 0
                return 1

        if action == 'LEFT':
            if not ( is_bomb_dangerous(bomb_coord, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_coord, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                if crate_in_expl_range:
                    if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                        return 0
                return 1


    #case 2:  two bombs-------------------------------------------------------
    if number_of_dangerous_bombs == 2:

        #get the bombs which are not 'None'
        bomb_index = [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None]
        bomb_coord1 = dangerous_bombs[ bomb_index[0] ]
        bomb_coord2 = dangerous_bombs[ bomb_index[1] ]

        x1 = bomb_coord1[0]
        y1 = bomb_coord1[1]

        x2 = bomb_coord2[0]
        y2 = bomb_coord2[1]


        # the next direction to be farther from both bombs
        if action == 'UP':
            #1st: gets farther from at least ONE bomb,  2nd: doesnt get closer to other bomb
            #gets out of range from bomb 1
            if not ( is_bomb_dangerous(bomb_coord1, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_coord1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y2 - (agent_coord_y - 1)) < abs(y2 - agent_coord_y)):
                    #dont run into crate trap 
                    """
                    ____W_
                    _B__xC (moved right in this case)
                    ____W_
                    """
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1
            #gets away from bomb 2
            if not ( is_bomb_dangerous(bomb_coord2, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_coord2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y1 - (agent_coord_y - 1)) < abs(y1 - agent_coord_y)):
                    #dont run into crate trap 
                    """
                    ____W_
                    _B__xC (moved right in this case)
                    ____W_
                    """
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y - 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1


        if action == 'RIGHT':
            #gets out of range from bomb 1
            if not ( is_bomb_dangerous(bomb_coord1, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_coord1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(x2 - (agent_coord_x + 1)) < abs(x2 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1
            #gets out of range from bomb 2
            if not ( is_bomb_dangerous(bomb_coord2, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_coord2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y][agent_coord_x+1] == 0 ):
                #doesnt get closer to bomb1
                if not (abs(x1 - (agent_coord_x + 1)) < abs(x1 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x + 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1


        if action == 'DOWN':
            #gets out of range from bomb 1
            if not ( is_bomb_dangerous(bomb_coord1, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_coord1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y2 - (agent_coord_y + 1)) < abs(y2 - agent_coord_y)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1
            #gets out of range from bomb 2
            if not ( is_bomb_dangerous(bomb_coord2, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_coord2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y+1][agent_coord_x] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(y1 - (agent_coord_y + 1)) < abs(y1 - agent_coord_y)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_y - (agent_coord_y + 1)) and abs(crate_x - agent_coord_x) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y+1][agent_coord_x+1] != 0 ):
                            return 0
                    return 1


        if action == 'LEFT':
            #gets out of range from bomb 1
            if not ( is_bomb_dangerous(bomb_coord1, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_coord1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                #doesnt get closer to bomb2
                if not (abs(x2 - (agent_coord_x - 1)) < abs(x2 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                            return 0
                    return 1
            #gets out of range from bomb 2
            if not ( is_bomb_dangerous(bomb_coord2, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_coord2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_y][agent_coord_x-1] == 0 ):
                #doesnt get closer to bomb1
                if not (abs(x1 - (agent_coord_x - 1)) < abs(x1 - agent_coord_x)):
                    #avoid crate trap
                    if crate_in_expl_range:
                        if abs(crate_x - (agent_coord_x - 1)) and abs(crate_y - agent_coord_y) == 0 and ( game_state['field'][agent_coord_y+1][agent_coord_x-1] != 0 ) and ( game_state['field'][agent_coord_y-1][agent_coord_x-1] != 0 ):
                            return 0
                    return 1



    #case 3:  three bombs (hell no, I aint gonna do this ... good luck)---------------------------------------



    return 0





def runs_into_explosion(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]


    # the next direction to be farther from the bomb
    if action == 'UP':
        if game_state['explosion_map'][agent_coord_y-1][agent_coord_x] != 0 :
            return 1

    if action == 'RIGHT':
        if game_state['explosion_map'][agent_coord_y][agent_coord_x+1] != 0 :
            return 1

    if action == 'DOWN':
        if game_state['explosion_map'][agent_coord_y+1][agent_coord_x] != 0 :
            return 1

    if action == 'LEFT':
        if game_state['explosion_map'][agent_coord_y][agent_coord_x-1] != 0 :
            return 1


    return 0



def runs_into_bomb_range_without_dying(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #generate a 'bomb_locations' list like the coin_locations, so like:  [(x,y), (x,y), (x,y)]
    bomb_locations = [item[0] for item in game_state['bombs']]
    bomb_infos = game_state['bombs']


    closest_bomb1 = (None, 0)
    closest_bomb2 = (None, 0)
    closest_bomb3 = (None, 0)
    closest_bomb4 = (None, 0)


    # find the 3 closest bombs
    if len(bomb_locations) != 0:
        #sort 'bomb_infos' list according to cartesian distance to agent
        bomb_infos_sorted = sorted( bomb_infos, key = lambda coord: np.linalg.norm([coord[0][0] - agent_coord_x, coord[0][1] - agent_coord_y] ) )

        if len(bomb_infos) == 1:
            closest_bomb1 = bomb_infos_sorted[0]
        
        elif len(bomb_infos) == 2:
            closest_bomb1 = bomb_infos_sorted[0]
            closest_bomb2 = bomb_infos_sorted[1]

        elif len(bomb_infos) == 3:
            closest_bomb1 = bomb_infos_sorted[0]
            closest_bomb2 = bomb_infos_sorted[1]
            closest_bomb3 = bomb_infos_sorted[2]
        
        elif len(bomb_infos) >= 3:
            closest_bomb1 = bomb_infos_sorted[0]
            closest_bomb2 = bomb_infos_sorted[1]
            closest_bomb3 = bomb_infos_sorted[2]
            closest_bomb4 = bomb_infos_sorted[3]

    closest_bombs = [closest_bomb1, closest_bomb2, closest_bomb3, closest_bomb4]

    #get coordinates
    length = len([i for i in range(len(closest_bombs)) if closest_bombs[i][0] != None])
    #only look at existing bombs
    closest_bombs = closest_bombs[:length]


    if action == 'UP':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if not (1 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1


    if action == 'RIGHT':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if not (1 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1

    if action == 'DOWN':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if not (1 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1

    if action == 'LEFT':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if not (1 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1



    return 0



#exactly the same as 'runs_into_bomb_range_without_dying', but remove the 'not' in the '1 in ...' condition
def runs_into_bomb_range_with_dying(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #generate a 'bomb_locations' list like the coin_locations, so like:  [(x,y), (x,y), (x,y)]
    bomb_locations = [item[0] for item in game_state['bombs']]
    bomb_infos = game_state['bombs']


    closest_bomb1 = (None, 0)
    closest_bomb2 = (None, 0)
    closest_bomb3 = (None, 0)
    closest_bomb4 = (None, 0)


    # find the 3 closest bombs
    if len(bomb_locations) != 0:
        #sort 'bomb_infos' list according to cartesian distance to agent
        bomb_infos_sorted = sorted( bomb_infos, key = lambda coord: np.linalg.norm([coord[0][0] - agent_coord_x, coord[0][1] - agent_coord_y] ) )

        if len(bomb_infos) == 1:
            closest_bomb1 = bomb_infos_sorted[0]
        
        elif len(bomb_infos) == 2:
            closest_bomb1 = bomb_infos_sorted[0]
            closest_bomb2 = bomb_infos_sorted[1]

        elif len(bomb_infos) == 3:
            closest_bomb1 = bomb_infos_sorted[0]
            closest_bomb2 = bomb_infos_sorted[1]
            closest_bomb3 = bomb_infos_sorted[2]
        
        elif len(bomb_infos) >= 3:
            closest_bomb1 = bomb_infos_sorted[0]
            closest_bomb2 = bomb_infos_sorted[1]
            closest_bomb3 = bomb_infos_sorted[2]
            closest_bomb4 = bomb_infos_sorted[3]

    closest_bombs = [closest_bomb1, closest_bomb2, closest_bomb3, closest_bomb4]

    #get coordinates
    length = len([i for i in range(len(closest_bombs)) if closest_bombs[i][0] != None])
    #only look at existing bombs
    closest_bombs = closest_bombs[:length]


    if action == 'UP':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if (1 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1


    if action == 'RIGHT':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if (1 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1

    if action == 'DOWN':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if (1 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1

    if action == 'LEFT':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i[0], agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_y-1][agent_coord_x] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if (1 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1



    return 0



def bomb_dropped_although_not_possible(game_state, action, self):

    if action == 'BOMB':
        if ( game_state["self"][2] == False ):
            return 1

    return 0


def waited(game_state, action, self):

    if action == 'WAIT':
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
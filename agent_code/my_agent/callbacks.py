import os
import pickle
import random

import numpy as np
from collections import deque
from itertools import compress #for 'find_planted_bombs_in_dangerous_range' function


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = 18
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
        self.random_or_choosen = 0 #only for logger (can be deleted at end)
        self.count_chosen_wall_crate_run = 0 #only for logger (can be deleted at end)
        self.count_chosen_action_loop = 0 #only for logger (can be deleted at end)
        self.count_crate_trap = 0 #only for logger (can be deleted at end)
        self.count_runs_into_explosion = 0 #only for logger (can be deleted at end)
        self.count_runs_into_bomb_range_no_dying = 0 #only for logger (can be deleted at end)
        self.count_runs_into_bomb_range_with_dying = 0 #only for logger (can be deleted at end)
        self.count_advanced_crate_trap = 0 #only for logger (can be deleted at end)


        self.taken_action = None 


        self.model = weights
    else:
        self.former_action = deque(maxlen=ACTION_HISTORY_SIZE)
        self.former_state = deque(maxlen=STATE_HISTORY_SIZE)
        self.action_loop_result_before_taken_action = 0
        self.a = 0 #for action loop
        self.random_or_choosen = 0 #only for logger (can be deleted at end)
        self.count_chosen_wall_crate_run = 0 #only for logger (can be deleted at end)
        self.count_chosen_action_loop = 0 #only for logger (can be deleted at end)
        self.count_crate_trap = 0 #only for logger (can be deleted at end)
        self.count_runs_into_explosion = 0 #only for logger (can be deleted at end)
        self.count_runs_into_bomb_range_no_dying = 0 #only for logger (can be deleted at end)
        self.count_runs_into_bomb_range_with_dying = 0 #only for logger (can be deleted at end)
        self.count_advanced_crate_trap = 0 #only for logger (can be deleted at end)

        
        self.taken_action = None 


        self.logger.info("LOADING MODEL FROM FILE")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


#done
def act(self, game_state: dict) -> str:

    #for logger
    self.logger.info(f'\n------------------------------------ Step: {game_state["step"]}')
    


    #Explore
    if self.train and random.random() < self.epsilon:
        action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0]) #PHASE 1 (No crates yet)
        
        self.random_or_choosen = 1 #only for logger (can be deleted at end)

      
        
    
    #Exploit (choose best action)
    else:

        weights = self.model
        features_for_action_1 = state_to_features(game_state, ACTIONS[0], self)
        features_for_action_2 = state_to_features(game_state, ACTIONS[1], self)
        features_for_action_3 = state_to_features(game_state, ACTIONS[2], self)
        features_for_action_4 = state_to_features(game_state, ACTIONS[3], self)
        features_for_action_5 = state_to_features(game_state, ACTIONS[4], self)
        features_for_action_6 = state_to_features(game_state, ACTIONS[5], self)


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
        
        self.random_or_choosen = 2 #only for logger (can be deleted at end)

       

    #to store taken action
    self.taken_action = action


    #for logger
    if len(self.former_action) == 4:
        self.logger.info(f"Last 4 actions without current action:  {self.former_action}") #deactivate if u dont wanna see
        self.logger.info(f"UP: {state_to_features(game_state, ACTIONS[0], self)[2]}, RIGHT: {state_to_features(game_state, ACTIONS[1], self)[2]}, DOWN: {state_to_features(game_state, ACTIONS[2], self)[2]}, LEFT: {state_to_features(game_state, ACTIONS[3], self)[2]}")

    #for logger
    if self.random_or_choosen == 2:
        self.logger.info(f"CHOSEN ACTION: {self.taken_action}")
    elif self.random_or_choosen == 1:
        self.logger.info(f"RANDOM ACTION: {self.taken_action}")


    self.former_action.append(self.taken_action) #CHANGING POINT FOR ACTION LOOP FUNCTION - FUTURE


    #for logger 
    if len(self.former_action) == 4:
        self.logger.info(f"Last 4 actions with current action:  {self.former_action}") #deactivate if u dont wanna see


        if (self.former_action[0] == self.former_action[2]) and (self.former_action[1] == self.former_action[3]) and (self.former_action[0] != self.former_action[1]) and (self.former_action[2] != self.former_action[3]):
            #up-down-up-down
            if self.former_action[0] in ['UP', 'DOWN']:
                if self.former_action[1] in ['UP', 'DOWN']:
                    if self.former_action[3] == self.former_action[1]:
                        self.logger.info(f"You are in a loop!")
                    #in case you want to disable 'up-down-up-wait' loop
                    # if action == 'WAIT':
                    #     self.action_loop_result_before_taken_action = 1
                    #     return -1                   

            #left-right-left-right
            if self.former_action[0] in ['LEFT', 'RIGHT']:
                if self.former_action[1] in ['LEFT', 'RIGHT']:
                    if self.former_action[3] == self.former_action[1]:
                        self.logger.info(f"You are in a loop!")
                    # if action == 'WAIT':
                    #     self.action_loop_result_before_taken_action = 1
                    #     return -1  



    self.former_action.appendleft(0) #CHANGING POINT FOR ACTION LOOP FUNCTION - NORMAL

    #to store self.action_loop_result_before_taken_action
    state_to_features(game_state, self.taken_action, self)

    self.logger.info(f"Var is_loop in func: {self.action_loop_result_before_taken_action}")


    self.former_action.append(self.taken_action) #CHANGING POINT FOR ACTION LOOP FUNCTION - FUTURE

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
    i = get_away_from_bomb(   find_planted_bombs_in_dangerous_range(game_state, action, self)  ,   game_state, action, self)
    j = get_out_of_bomb_range(   find_planted_bombs_in_dangerous_range(game_state, action, self)   ,    game_state, action, self)
    k = goes_towards_crate_trap(   find_planted_bombs_in_dangerous_range(game_state, action, self)   ,    game_state, action, self)
    l = bomb_dropped_although_not_possible(game_state, action, self)

    m = runs_into_explosion(game_state, action, self)
    n = runs_into_bomb_range_without_dying(game_state, action, self)
    o = runs_into_bomb_range_with_dying(game_state, action, self)

    p = goes_towards_dangerous_bomb(   find_planted_bombs_in_dangerous_range(game_state, action, self)   ,    game_state, action, self)
    q = move_into_advanced_crate_trap(   find_planted_bombs_in_dangerous_range(game_state, action, self)   ,    game_state, action, self)

    at_end2 = waited(game_state, action, self)


    

    # #only for coin heaven stage
    # at_end1 = bomb_dropped(game_state, action, self)
    # at_end2 = waited(game_state, action, self)


    return np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, at_end2]) 


#done
def f0():
    return 1


#done
def runs_into_wall_crate(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #CAUTION x and y are have to be accessed as follows in the 'field' variable
    moved_up = game_state['field'][agent_coord_x][agent_coord_y-1] #Up    (definitly correct)
    moved_ri = game_state['field'][agent_coord_x+1][agent_coord_y] #Right (definitly correct)
    moved_do = game_state['field'][agent_coord_x][agent_coord_y+1] #Down  (definitly correct)
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


#done
def action_loop(game_state, action, self):

    #if in action loop
    if len(self.former_action) == 4:
        if (self.former_action[1] == self.former_action[3]) and (self.former_action[2] == action) and (self.former_action[1] != self.former_action[2]) and (self.former_action[3] != action):
            #up-down-up-down
            if self.former_action[1] in ['UP', 'DOWN']:
                if self.former_action[2] in ['UP', 'DOWN']:
                    if action == self.former_action[2]:
                        self.action_loop_result_before_taken_action = 1
                        return -1
                    #in case you want to disable 'up-down-up-wait' loop
                    # if action == 'WAIT':
                    #     self.action_loop_result_before_taken_action = 1
                    #     return -1                   

            #left-right-left-right
            if self.former_action[1] in ['LEFT', 'RIGHT']:
                if self.former_action[2] in ['LEFT', 'RIGHT']:
                    if action == self.former_action[2]:
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


#done (fixed x - y coord)
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


    # self.logger.info(f"Coordinates of coins: \n {coin_locations}")
    # self.logger.info(f"Coordinates of closest coin: {closest_coin}")


    # the next direction to be closer to the closest coin
    if closest_coin is not None:
        
        x, y = closest_coin

        if action == 'UP':
            #check if distance along changing axis is reduced and there is no wall or crate on the field u go to
            if abs(y - (agent_coord_y - 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
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
            if abs(x - (agent_coord_x + 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
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
            if abs(y - (agent_coord_y + 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
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
            if abs(x - (agent_coord_x - 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
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

#done (fixed x - y coord)
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
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                return 1

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                return 1

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                return 1

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                return 1


    return 0



# helper function    done (x and y correct)
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

#done (x and y correct)
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

#hopefully done (x and y fixed?)
def runs_towards_closest_crate_but_not_wall_or_crate(closest_crate, game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    # the next direction to be closer to the closest crate
    if closest_crate is not None:
        
        x, y = closest_crate


        if action == 'UP':
            #check if distance along changing axis is reduced and there is no wall or crate on the field u go to
            if abs(y - (agent_coord_y - 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
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
            if abs(x - (agent_coord_x + 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                if abs(x - (agent_coord_x + 1)) == 0 and abs(y - agent_coord_y) == 0:
                    return 0
                if ((agent_coord_x + 1) % 2) != 0: 
                    return 1
                else:
                    if abs(x - (agent_coord_x + 1)) != 0 :
                        return 1
      

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) < abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                if abs(y - (agent_coord_y + 1)) == 0 and abs(x - agent_coord_x) == 0:
                    return 0                
                if ((agent_coord_y + 1) % 2) != 0: 
                    return 1
                else:
                    if abs(y - (agent_coord_y + 1)) != 0 :
                        return 1
              

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) < abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                if abs(x - (agent_coord_x - 1)) == 0 and abs(y - agent_coord_y) == 0:
                    return 0                
                if ((agent_coord_x - 1) % 2) != 0: 
                    return 1
                else:
                    if abs(x - (agent_coord_x - 1)) != 0 :
                        return 1


    return 0

#hopefully done (x and y fixed?)
def runs_away_from_closest_crate_but_not_wall_or_crate(closest_crate, game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    # the next direction to be closer to the closest crate
    if closest_crate is not None:
        
        x, y = closest_crate


        if action == 'UP':
            #check if distance along changing axis is reduced and there is no wall or crate on the field u go to
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                #if agent would move into crate dont give reward
                if abs(y - (agent_coord_y - 1)) == 0 and abs(x - agent_coord_x) == 0:
                    return 0
                else:
                    return 1


        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                if abs(x - (agent_coord_x + 1)) == 0 and abs(y - agent_coord_y) == 0:
                    return 0
                else:
                    return 1
      

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                if abs(y - (agent_coord_y + 1)) == 0 and abs(x - agent_coord_x) == 0:
                    return 0                
                else:
                    return 1
              

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                if abs(x - (agent_coord_x - 1)) == 0 and abs(y - agent_coord_y) == 0:
                    return 0                
                else:
                    return 1


    return 0


#done (x and y fixed)
#helper helper function
def is_bomb_dangerous(bomb, agent_coord_x, agent_coord_y):

    if bomb[0] is not None:

        #agent in y range of explosion
        if abs(bomb[0][1] - agent_coord_y) <= 3 and abs(bomb[0][0] - agent_coord_x) == 0:
            #no wall in between
            if (agent_coord_x % 2) != 0 :
                return True
        
        #agent in x range of explosion
        if abs(bomb[0][0] - agent_coord_x) <= 3 and abs(bomb[0][1] - agent_coord_y) == 0:
            #no wall in between
            if (agent_coord_y % 2) != 0 :
                return True

        #agent directly on bomb
        if abs(bomb[0][0] - agent_coord_x) == 0 and abs(bomb[0][1] - agent_coord_y) == 0:
            return True       

    return False




#hopefully done (x and y fixed)
#helper function
def find_planted_bombs_in_dangerous_range(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    #generate a 'bomb_infos' list similar to the coin_locations, but like:  [((x,y),t), ((x,y),t), ((x,y),t)]
    bomb_infos = game_state["bombs"]

    #only keep bombs for which agent is in explosion range
    filter_criteria = [is_bomb_dangerous(x, agent_coord_x, agent_coord_y) for x in game_state["bombs"]]
    bomb_infos = list(compress(bomb_infos, filter_criteria))

    closest_bomb1 = (None, 0)
    closest_bomb2 = (None, 0)
    closest_bomb3 = (None, 0)
    closest_dist = 100

    # find the 3 closest bombs
    if len(bomb_infos) != 0:
        
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
        if is_bomb_dangerous(closest_bombs[0], agent_coord_x, agent_coord_y):
            number_of_dangerous_bombs += 1
        else:
            closest_bomb1 = (None, 0)

    if closest_bombs[1][0] is not None:
        if is_bomb_dangerous(closest_bombs[1], agent_coord_x, agent_coord_y):
            number_of_dangerous_bombs += 1
        else:
            closest_bomb2 = (None, 0)

    if closest_bombs[2][0] is not None:
        if is_bomb_dangerous(closest_bombs[2], agent_coord_x, agent_coord_y):
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


#hopefully done (x and y ficed)
#here we would still need to solve crate trap problem for 2 crates (only solved for one)
# arg: 'dangerous_bombs, number_of_dangerous_bombs'  is  'find_planted_bombs_in_dangerous_range(game_state, action, self)'
# arg: 'closest_crate' is 'find_closest_crates(game_state, action, self)'
def get_away_from_bomb(dangerous_bombs_and_number_of_dangerous_bombs, game_state, action, self):

    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]




    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]

        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y 


        # the next direction to be farther from the bomb
        if action == 'UP':
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                #avoid crate trap
                """
                ____W_
                _B__xC  (moved right in this case)
                ____W_
                """
                #if agent moved to free lane, crate problem cant occure
                if ((agent_coord_y - 1) % 2) != 0:
                    return 1
                #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate above
                elif ((agent_coord_y - 1) % 2) == 0 and ((agent_coord_y - 1) != 0) :
                    if game_state['field'][agent_coord_x][agent_coord_y-2] != 0:
                        return 0

                return 1

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                if ((agent_coord_x + 1) % 2) != 0:
                    return 1
                elif ((agent_coord_x + 1) % 2) == 0 and ((agent_coord_x + 1) != len(game_state['field'])-1):
                    if game_state['field'][agent_coord_x+2][agent_coord_y] != 0:
                        return 0
                        
                return 1

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                if ((agent_coord_y + 1) % 2) != 0:
                    return 1
                elif ((agent_coord_y + 1) % 2) == 0 and ((agent_coord_y + 1) != len(game_state['field'][:][0])-1) :
                    if game_state['field'][agent_coord_x][agent_coord_y+2] != 0:
                        return 0
                        
                return 1

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                if ((agent_coord_x - 1) % 2) != 0:
                    return 1
                elif ((agent_coord_x - 1) % 2) == 0 and ((agent_coord_x - 1) != 0):
                    if game_state['field'][agent_coord_x-2][agent_coord_y] != 0:
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

            #if agent IS in free lane 
            if ((agent_coord_y) % 2) != 0:

                #1st: gets farther from at least ONE bomb,  2nd: doesnt get closer to other bomb
                #gets away from bomb 1
                if abs(y1 - (agent_coord_y - 1)) > abs(y1 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                    #doesnt get closer to bomb2
                    if abs(y2 - (agent_coord_y - 1)) >= abs(y2 - agent_coord_y):
                        #avoid crate trap
                        """
                        ____W_
                        _B__xC  (moved right in this case)
                        ____W_
                        """
                        #if agent MOVED to free lane (cannot be true if he already was in free lane and moved)
                        if ((agent_coord_y - 1) % 2) != 0:
                            return 1
                        #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate above
                        elif ((agent_coord_y - 1) % 2) == 0 and ((agent_coord_y - 1) != 0) :
                            if game_state['field'][agent_coord_x][agent_coord_y-2] != 0:
                                return 0

                        return 1
                    

                #gets away from bomb 2
                if abs(y2 - (agent_coord_y - 1)) > abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                    #doesnt get closer to bomb1
                    if abs(y1 - (agent_coord_y - 1)) >= abs(y1 - agent_coord_y):
                        #avoid crate trap
                        """
                        ____W_
                        _B__xC  (moved right in this case)
                        ____W_
                        """
                        #if agent MOVED to free lane (cannot be true if he already was in free lane and moved)
                        if ((agent_coord_y - 1) % 2) != 0:
                            return 1
                        #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate above
                        elif ((agent_coord_y - 1) % 2) == 0 and ((agent_coord_y - 1) != 0) :
                            if game_state['field'][agent_coord_x][agent_coord_y-2] != 0:
                                return 0

                        return 1


            #if agent IS in problem lane  |_|_|x|_|_|
            if ((agent_coord_y) % 2) == 0:
                #if both bombs are above agent return 0, else return 1
                if y1 <= agent_coord_y and y2 <= agent_coord_y:
                    return 0
                else:
                    return 1




        if action == 'RIGHT':

            if ((agent_coord_x) % 2) != 0:

                if abs(x1 - (agent_coord_x + 1)) > abs(x1 - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                    if abs(x2 - (agent_coord_x + 1)) >= abs(x2 - agent_coord_x):
                        if ((agent_coord_x + 1) % 2) != 0:
                            return 1
                        elif ((agent_coord_x + 1) % 2) == 0 and ((agent_coord_x + 1) != len(game_state['field'])-1):
                            if game_state['field'][agent_coord_x+2][agent_coord_y] != 0:
                                return 0

                        return 1
                    

                if abs(x2 - (agent_coord_x + 1)) > abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                    if abs(x1 - (agent_coord_x + 1)) >= abs(x1 - agent_coord_x):
                        if ((agent_coord_x + 1) % 2) != 0:
                            return 1
                        elif ((agent_coord_x + 1) % 2) == 0 and ((agent_coord_x + 1) != len(game_state['field'])-1):
                            if game_state['field'][agent_coord_x+2][agent_coord_y] != 0:
                                return 0

                        return 1


            if ((agent_coord_x) % 2) == 0:
                if x1 >= agent_coord_x and x2 >= agent_coord_x:
                    return 0
                else:
                    return 1




        if action == 'DOWN':

            if ((agent_coord_y) % 2) != 0:

                if abs(y1 - (agent_coord_y + 1)) > abs(y1 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                    if abs(y2 - (agent_coord_y + 1)) >= abs(y2 - agent_coord_y):
                        if ((agent_coord_y + 1) % 2) != 0:
                            return 1
                        elif ((agent_coord_y + 1) % 2) == 0 and ((agent_coord_y + 1) != len(game_state['field'][:][0])-1) :
                            if game_state['field'][agent_coord_x][agent_coord_y+2] != 0:
                                return 0

                        return 1
                    

                if abs(y2 - (agent_coord_y + 1)) > abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                    if abs(y1 - (agent_coord_y + 1)) >= abs(y1 - agent_coord_y):
                        if ((agent_coord_y + 1) % 2) != 0:
                            return 1
                        elif ((agent_coord_y + 1) % 2) == 0 and ((agent_coord_y + 1) != len(game_state['field'][:][0])-1) :
                            if game_state['field'][agent_coord_x][agent_coord_y+2] != 0:
                                return 0

                        return 1


            if ((agent_coord_y) % 2) == 0:
                if y1 >= agent_coord_y and y2 >= agent_coord_y:
                    return 0
                else:
                    return 1





        if action == 'LEFT':

            if ((agent_coord_x) % 2) != 0:

                if abs(x1 - (agent_coord_x - 1)) > abs(x1 - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                    if abs(x2 - (agent_coord_x - 1)) >= abs(x2 - agent_coord_x):
                        if ((agent_coord_x - 1) % 2) != 0:
                            return 1
                        elif ((agent_coord_x - 1) % 2) == 0 and ((agent_coord_x - 1) != 0):
                            if game_state['field'][agent_coord_x-2][agent_coord_y] != 0:
                                return 0

                        return 1
                    

                if abs(x2 - (agent_coord_x - 1)) > abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                    if abs(x1 - (agent_coord_x - 1)) >= abs(x1 - agent_coord_x):
                        if ((agent_coord_x - 1) % 2) != 0:
                            return 1
                        elif ((agent_coord_x - 1) % 2) == 0 and ((agent_coord_x - 1) != 0):
                            if game_state['field'][agent_coord_x-2][agent_coord_y] != 0:
                                return 0

                        return 1


            if ((agent_coord_x) % 2) == 0:
                if x1 <= agent_coord_x and x2 <= agent_coord_x:
                    return 0
                else:
                    return 1



    #case 3:  three bombs (hell no, I aint gonna do this ... good luck)---------------------------------------



    return 0




#hopefully done (x and y ficed)
#here we would still need to solve crate trap problem for 2 crates (only solved for one)
#exactly the same as 'get_away_from_bomb' only switch all return 0 and 1 values (except the final one)
def goes_towards_crate_trap(dangerous_bombs_and_number_of_dangerous_bombs, game_state, action, self):

    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]




    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]

        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y 


        # the next direction to be farther from the bomb
        if action == 'UP':
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                #avoid crate trap
                """
                ____W_
                _B__xC  (moved right in this case)
                ____W_
                """
                #if agent moved to free lane crate problem cant occure
                if ((agent_coord_y - 1) % 2) != 0:
                    return 0
                #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate above
                elif ((agent_coord_y - 1) % 2) == 0 and ((agent_coord_y - 1) != 0) :
                    if game_state['field'][agent_coord_x][agent_coord_y-2] != 0:
                        return 1

                return 0

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                if ((agent_coord_x + 1) % 2) != 0:
                    return 0
                elif ((agent_coord_x + 1) % 2) == 0 and ((agent_coord_x + 1) != len(game_state['field'])-1):
                    if game_state['field'][agent_coord_x+2][agent_coord_y] != 0:
                        return 1
                        
                return 0

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                if ((agent_coord_y + 1) % 2) != 0:
                    return 0
                elif ((agent_coord_y + 1) % 2) == 0 and ((agent_coord_y + 1) != len(game_state['field'][:][0])-1) :
                    if game_state['field'][agent_coord_x][agent_coord_y+2] != 0:
                        return 1
                        
                return 0

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                if ((agent_coord_x - 1) % 2) != 0:
                    return 0
                elif ((agent_coord_x - 1) % 2) == 0 and ((agent_coord_x - 1) != 0):
                    if game_state['field'][agent_coord_x-2][agent_coord_y] != 0:
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

            #if agent IS in free lane 
            if ((agent_coord_y) % 2) != 0:

                #1st: gets farther from at least ONE bomb,  2nd: doesnt get closer to other bomb
                #gets away from bomb 1
                if abs(y1 - (agent_coord_y - 1)) > abs(y1 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                    #doesnt get closer to bomb2
                    if abs(y2 - (agent_coord_y - 1)) >= abs(y2 - agent_coord_y):
                        #avoid crate trap
                        """
                        ____W_
                        _B__xC  (moved right in this case)
                        ____W_
                        """
                        #if agent MOVED to free lane (cannot be true if he already was in free lane and moved)
                        if ((agent_coord_y - 1) % 2) != 0:
                            return 0
                        #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate above
                        elif ((agent_coord_y - 1) % 2) == 0 and ((agent_coord_y - 1) != 0) :
                            if game_state['field'][agent_coord_x][agent_coord_y-2] != 0:
                                return 1

                        return 0
                    

                #gets away from bomb 2
                if abs(y2 - (agent_coord_y - 1)) > abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                    #doesnt get closer to bomb1
                    if abs(y1 - (agent_coord_y - 1)) >= abs(y1 - agent_coord_y):
                        #avoid crate trap
                        """
                        ____W_
                        _B__xC  (moved right in this case)
                        ____W_
                        """
                        #if agent MOVED to free lane (cannot be true if he already was in free lane and moved)
                        if ((agent_coord_y - 1) % 2) != 0:
                            return 0
                        #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate above
                        elif ((agent_coord_y - 1) % 2) == 0 and ((agent_coord_y - 1) != 0) :
                            if game_state['field'][agent_coord_x][agent_coord_y-2] != 0:
                                return 1

                        return 0


            #if agent IS in problem lane  |_|_|x|_|_| -------> there CANT be a crate trap!!!
            if ((agent_coord_y) % 2) == 0:
                return 0




        if action == 'RIGHT':

            if ((agent_coord_x) % 2) != 0:

                if abs(x1 - (agent_coord_x + 1)) > abs(x1 - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                    if abs(x2 - (agent_coord_x + 1)) >= abs(x2 - agent_coord_x):
                        if ((agent_coord_x + 1) % 2) != 0:
                            return 0
                        elif ((agent_coord_x + 1) % 2) == 0 and ((agent_coord_x + 1) != len(game_state['field'])-1):
                            if game_state['field'][agent_coord_x+2][agent_coord_y] != 0:
                                return 1

                        return 0
                    

                if abs(x2 - (agent_coord_x + 1)) > abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                    if abs(x1 - (agent_coord_x + 1)) >= abs(x1 - agent_coord_x):
                        if ((agent_coord_x + 1) % 2) != 0:
                            return 0
                        elif ((agent_coord_x + 1) % 2) == 0 and ((agent_coord_x + 1) != len(game_state['field'])-1):
                            if game_state['field'][agent_coord_x+2][agent_coord_y] != 0:
                                return 1

                        return 0


            if ((agent_coord_x) % 2) == 0:
                return 0




        if action == 'DOWN':

            if ((agent_coord_y) % 2) != 0:

                if abs(y1 - (agent_coord_y + 1)) > abs(y1 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                    if abs(y2 - (agent_coord_y + 1)) >= abs(y2 - agent_coord_y):
                        if ((agent_coord_y + 1) % 2) != 0:
                            return 0
                        elif ((agent_coord_y + 1) % 2) == 0 and ((agent_coord_y + 1) != len(game_state['field'][:][0])-1) :
                            if game_state['field'][agent_coord_x][agent_coord_y+2] != 0:
                                return 1

                        return 0
                    

                if abs(y2 - (agent_coord_y + 1)) > abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                    if abs(y1 - (agent_coord_y + 1)) >= abs(y1 - agent_coord_y):
                        if ((agent_coord_y + 1) % 2) != 0:
                            return 0
                        elif ((agent_coord_y + 1) % 2) == 0 and ((agent_coord_y + 1) != len(game_state['field'][:][0])-1) :
                            if game_state['field'][agent_coord_x][agent_coord_y+2] != 0:
                                return 1

                        return 0


            if ((agent_coord_y) % 2) == 0:
                return 0





        if action == 'LEFT':

            if ((agent_coord_x) % 2) != 0:

                if abs(x1 - (agent_coord_x - 1)) > abs(x1 - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                    if abs(x2 - (agent_coord_x - 1)) >= abs(x2 - agent_coord_x):
                        if ((agent_coord_x - 1) % 2) != 0:
                            return 0
                        elif ((agent_coord_x - 1) % 2) == 0 and ((agent_coord_x - 1) != 0):
                            if game_state['field'][agent_coord_x-2][agent_coord_y] != 0:
                                return 1

                        return 0
                    

                if abs(x2 - (agent_coord_x - 1)) > abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                    if abs(x1 - (agent_coord_x - 1)) >= abs(x1 - agent_coord_x):
                        if ((agent_coord_x - 1) % 2) != 0:
                            return 0
                        elif ((agent_coord_x - 1) % 2) == 0 and ((agent_coord_x - 1) != 0):
                            if game_state['field'][agent_coord_x-2][agent_coord_y] != 0:
                                return 1

                        return 0


            if ((agent_coord_x) % 2) == 0:
                return 0



    #case 3:  three bombs (hell no, I aint gonna do this ... good luck)---------------------------------------



    return 0









#hopefully done (x and y ficed)
#here we would still need to solve crate trap problem for 2 crates (only solved for one)
# arg: 'dangerous_bombs, number_of_dangerous_bombs'  is  'find_planted_bombs_in_dangerous_range(game_state, action, self)'
def get_out_of_bomb_range(dangerous_bombs_and_number_of_dangerous_bombs, game_state, action, self):
    
    #only coordinates [(x,y), (x,y)]
    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]]
    #all info [((x,y),t), ((x,y),t)]
    dangerous_bombs_info = [item for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 

    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]




    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        #get only coordinates (x,y) (no brackets)
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]
        #get all info [((x,y),t)]   (w/ brackets)
        bomb_info = [dangerous_bombs_info[i] for i in [i for i in range(len(dangerous_bombs_info)) if dangerous_bombs_info[i][0] != None]]

        #get the only existing bomb ((x,y),t)  (no brackets)
        bomb_arg = bomb_info[0]
        
        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y 



        # the direction which would move agent out of explosion range (if there is no wall or crate)
        # (moving out of range but into a crate trap has no special consideration in this func BUT through 'goes_towards_crate_trap' 
        # func. agent will still be more likely to move out of range AND not into crate trap)
        if action == 'UP':
            if not ( is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                return 1

        if action == 'RIGHT':
            if not ( is_bomb_dangerous(bomb_arg, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                return 1

        if action == 'DOWN':
            if not ( is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                return 1

        if action == 'LEFT':
            if not ( is_bomb_dangerous(bomb_arg, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                return 1

    




    #case 2:  two bombs-------------------------------------------------------
    if number_of_dangerous_bombs == 2:

        #get the bombs which are not 'None'
        #get the indices [0,1] of not 'None' bombs in 'dangerous_bombs' list
        bomb_index = [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None]
        #get the indices [0,1] of not 'None' bombs in 'dangerous_bombs_info' list
        bomb_index_info = [i for i in range(len(dangerous_bombs_info)) if dangerous_bombs_info[i] != None]

        #get list of coordinates [(x,y), (x,y)] of bombs w/out 'None'
        bomb_coords = [dangerous_bombs[i] for i in bomb_index]
        #get list of all infos [((x,y),t), ((x,y),t)] of bombs w/out 'None'
        bomb_info = [dangerous_bombs_info[i] for i in bomb_index_info]

        #choose the 2 existing bombs coords (x,y)
        bomb_coord1 = bomb_coords[0]
        bomb_coord2 = bomb_coords[1]

        #choose the existing bombs infos ((x,y),t)
        bomb_arg1 = bomb_info[0]
        bomb_arg2 = bomb_info[1]
        


        x1 = bomb_coord1[0]
        y1 = bomb_coord1[1]

        x2 = bomb_coord2[0]
        y2 = bomb_coord2[1]



        # the next direction to be farther from both bombs
        if action == 'UP':

            #if agent moved to free lane, crate problem cant occure
            if ((agent_coord_y - 1) % 2) != 0:
                #check if agent can exit at least one bomb range
                if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                    return 1
                if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                    return 1

            #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate above
            else:
                #move perpendicular to both bomb lines
                if y1 == y2 :
                    if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                        return 1
                    if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                        return 1
                #move parallel to both bomb lines
                elif x1 == x2:
                    #only move up if both bombs are below you, you exit one range and dont run into trap
                    #both below you
                    if y1 >= agent_coord_y and y2 >= agent_coord_y: ########################
                        #you dont run into trap
                        if game_state['field'][agent_coord_x][agent_coord_y-1] != -1:
                            if game_state['field'][agent_coord_x][agent_coord_y-2] == 0:
                                #you exist one of the bombs
                                #exit bomb 1
                                if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                                    return 1
                                #exit bomb 2
                                if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                                    return 1

                #moves parallel to one bomb and perpendicular to other
                else:
                    #agent exits both bombs
                    if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                        if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                            return 1
                    #agent exits one bomb, doesnt run into trap and doesnt get closer to second bomb
                    #you dont run into trap
                    if game_state['field'][agent_coord_x][agent_coord_y-1] != -1:
                        if game_state['field'][agent_coord_x][agent_coord_y-2] == 0: 
                            #exit bomb1
                            if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                                #doesnt get closer to bomb2
                                if abs(y2 - (agent_coord_y - 1)) >= abs(y2 - agent_coord_y): 
                                    return 1
                            #exit bomb2
                            if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                                #doesnt get closer to bomb1
                                if abs(y1 - (agent_coord_y - 1)) >= abs(y1 - agent_coord_y): 
                                    return 1






        if action == 'RIGHT':
            #if agent moved to free lane, crate problem cant occure
            if ((agent_coord_x + 1) % 2) != 0:
                #check if agent can exit at least one bomb range
                if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                    return 1
                if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                    return 1

            #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate to the right
            else:
                #move perpendicular to both bomb lines
                if x1 == x2 :
                    if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                        return 1
                    if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                        return 1
                #move parallel to both bomb lines
                elif y1 == y2:
                    #only move up if both bombs are left of you, you exit one range and dont run into trap
                    #both below you
                    if x1 <= agent_coord_x and x2 <= agent_coord_x: ####################
                        #you dont run into trap
                        if game_state['field'][agent_coord_x+1][agent_coord_y] != -1:
                            if game_state['field'][agent_coord_x+2][agent_coord_y] == 0:
                                #you exist one of the bombs
                                #exit bomb 1
                                if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                                    return 1
                                #exit bomb 2
                                if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                                    return 1

                #moves parallel to one bomb and perpendicular to other
                else:
                    #agent exits both bombs
                    if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                        if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                            return 1
                    #agent exits one bomb, doesnt run into trap and doesnt get closer to second bomb
                    #you dont run into trap
                    if game_state['field'][agent_coord_x+1][agent_coord_y] != -1:
                        if game_state['field'][agent_coord_x+2][agent_coord_y] == 0: 
                            #exit bomb1
                            if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                                #doesnt get closer to bomb2
                                if abs(x2 - (agent_coord_x + 1)) >= abs(x2 - agent_coord_x): 
                                    return 1
                            #exit bomb2
                            if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                                #doesnt get closer to bomb1
                                if abs(x1 - (agent_coord_x + 1)) >= abs(x1 - agent_coord_x): 
                                    return 1






        if action == 'DOWN':
            #if agent IS in problematic lane  |_|_|x|_|_| ------> no crate trap possible
            if ((agent_coord_y) % 2) == 0:
                #check if agent can exit at least one bomb range
                if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                    return 1
                if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                    return 1

            #if agent IS in free lane
            if ((agent_coord_y) % 2) != 0:
                #move perpendicular to both bomb lines
                if y1 == y2 :
                    if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                        return 1
                    if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                        return 1
                #move parallel to both bomb lines
                elif x1 == x2:
                    #only move up if both bombs are above you, you exit one range and dont run into trap
                    #both above you
                    if y1 <= agent_coord_y and y2 <= agent_coord_y: ########################
                        #you dont run into trap
                        if game_state['field'][agent_coord_x][agent_coord_y+1] != -1:
                            if game_state['field'][agent_coord_x][agent_coord_y+2] == 0:
                                #you exist one of the bombs
                                #exit bomb 1
                                if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                                    return 1
                                #exit bomb 2
                                if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                                    return 1

                #moves parallel to one bomb and perpendicular to other
                else:
                    #agent exits both bombs
                    if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                        if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                            return 1
                    #agent exits one bomb, doesnt run into trap and doesnt get closer to second bomb
                    #you dont run into trap
                    if game_state['field'][agent_coord_x][agent_coord_y+1] != -1:
                        if game_state['field'][agent_coord_x][agent_coord_y+2] == 0: 
                            #exit bomb1
                            if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                                #doesnt get closer to bomb2
                                if abs(y2 - (agent_coord_y + 1)) >= abs(y2 - agent_coord_y): 
                                    return 1
                            #exit bomb2
                            if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                                #doesnt get closer to bomb1
                                if abs(y1 - (agent_coord_y + 1)) >= abs(y1 - agent_coord_y): 
                                    return 1


            



        if action == 'LEFT':
            #if agent moved to free lane, crate problem cant occure
            if ((agent_coord_x - 1) % 2) != 0:
                #check if agent can exit at least one bomb range
                if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                    return 1
                if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                    return 1

            #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate to the right
            else:
                #move perpendicular to both bomb lines
                if x1 == x2 :
                    if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                        return 1
                    if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                        return 1
                #move parallel to both bomb lines
                elif y1 == y2:
                    #only move up if both bombs are right of you, you exit one range and dont run into trap
                    #both right to you
                    if x1 >= agent_coord_x and x2 >= agent_coord_x: ####################
                        #you dont run into trap
                        if game_state['field'][agent_coord_x-1][agent_coord_y] != -1:
                            if game_state['field'][agent_coord_x-2][agent_coord_y] == 0:
                                #you exist one of the bombs
                                #exit bomb 1
                                if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                                    return 1
                                #exit bomb 2
                                if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                                    return 1

                #moves parallel to one bomb and perpendicular to other
                else:
                    #agent exits both bombs
                    if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                        if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                            return 1
                    #agent exits one bomb, doesnt run into trap and doesnt get closer to second bomb
                    #you dont run into trap
                    if game_state['field'][agent_coord_x-1][agent_coord_y] != -1:
                        if game_state['field'][agent_coord_x-2][agent_coord_y] == 0: 
                            #exit bomb1
                            if not ( is_bomb_dangerous(bomb_arg1, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg1, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                                #doesnt get closer to bomb2
                                if abs(x2 - (agent_coord_x - 1)) >= abs(x2 - agent_coord_x): 
                                    return 1
                            #exit bomb2
                            if not ( is_bomb_dangerous(bomb_arg2, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg2, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                                #doesnt get closer to bomb1
                                if abs(x1 - (agent_coord_x - 1)) >= abs(x1 - agent_coord_x): 
                                    return 1






    #case 3:  three bombs (hell no, I aint gonna do this ... good luck)---------------------------------------



    return 0








#done (added)
# arg: 'dangerous_bombs_and_number_of_dangerous_bombs'  is  'find_planted_bombs_in_dangerous_range(game_state, action, self)'
def goes_towards_dangerous_bomb(dangerous_bombs_and_number_of_dangerous_bombs, game_state, action, self):

    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]




    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]

        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y 


        # the next direction to be closer to the dangerous bomb
        if action == 'UP':
            if abs(y - (agent_coord_y - 1)) <= abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                return 1

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) <= abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):         
                return 1

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) <= abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):                       
                return 1

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) <= abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):                        
                return 1

        if action == 'WAIT':
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


        if action == 'WAIT':
            return 1


        #if agent IS at |_|_|x|_|_| lane
        if ((agent_coord_y) % 2) == 0:
            if action == 'WAIT':
                return 1
        
        #if agent IS at |_|_|x|_|_| lane
        elif ((agent_coord_x) % 2) == 0:
            if action == 'WAIT':
                return 1

        #if agent in free cross
        else:
            if action == 'UP':
                #both bombs above you
                if abs(y1 - (agent_coord_y - 1)) <= abs(y1 - agent_coord_y) and abs(y2 - (agent_coord_y - 1)) <= abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                    return 1
                #one above, one below you
                if x1 == x2:
                    if (y1 < (agent_coord_y - 1)) and (y2 > agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                        return 1
                    if (y2 < (agent_coord_y - 1)) and (y1 > agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                        return 1
                #one left, one right
                elif y1 == y2:
                    return 0

                #one parallel, one perpendicular
                else:
                    #if you get closer to any of the 2 bombs agent is dumb
                    if abs(y1 - (agent_coord_y - 1)) <= abs(y1 - agent_coord_y) or abs(y2 - (agent_coord_y - 1)) <= abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                        return 1
                
            
            if action == 'RIGHT':
                #both bombs above you
                if abs(x1 - (agent_coord_x + 1)) <= abs(x1 - agent_coord_x) and abs(x2 - (agent_coord_x + 1)) <= abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                    return 1
                #one above, one below you
                if x1 == x2:
                    return 0
                #one left, one right
                elif y1 == y2:
                    if (x1 < (agent_coord_x + 1)) and (x2 > agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                        return 1
                    if (x2 < (agent_coord_x + 1)) and (x1 > agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                        return 1

                #one parallel, one perpendicular
                else:
                    #if you get closer to any of the 2 bombs agent is dumb
                    if abs(x1 - (agent_coord_x + 1)) <= abs(x1 - agent_coord_x) or abs(x2 - (agent_coord_x + 1)) <= abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                        return 1
            

            if action == 'DOWN':
                #both bombs above you
                if abs(y1 - (agent_coord_y + 1)) <= abs(y1 - agent_coord_y) and abs(y2 - (agent_coord_y + 1)) <= abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                    return 1
                #one above, one below you
                if x1 == x2:
                    if (y1 < (agent_coord_y + 1)) and (y2 > agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                        return 1
                    if (y2 < (agent_coord_y + 1)) and (y1 > agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                        return 1
                #one left, one right
                elif y1 == y2:
                    return 0

                #one parallel, one perpendicular
                else:
                    #if you get closer to any of the 2 bombs agent is dumb
                    if abs(y1 - (agent_coord_y + 1)) <= abs(y1 - agent_coord_y) or abs(y2 - (agent_coord_y + 1)) <= abs(y2 - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                        return 1
            

            if action == 'LEFT':
                #both bombs above you
                if abs(x1 - (agent_coord_x - 1)) <= abs(x1 - agent_coord_x) and abs(x2 - (agent_coord_x - 1)) <= abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                    return 1
                #one above, one below you
                if x1 == x2:
                    return 0
                #one left, one right
                elif y1 == y2:
                    if (x1 < (agent_coord_x - 1)) and (x2 > agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                        return 1
                    if (x2 < (agent_coord_x - 1)) and (x1 > agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                        return 1

                #one parallel, one perpendicular
                else:
                    #if you get closer to any of the 2 bombs agent is dumb
                    if abs(x1 - (agent_coord_x - 1)) <= abs(x1 - agent_coord_x) or abs(x2 - (agent_coord_x - 1)) <= abs(x2 - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                        return 1


    #case 3:  three bombs (hell no, I aint gonna do this ... good luck)---------------------------------------



    return 0





#done (added)
#helper function
def move_into_advanced_crate_trap_with_this_bomb(x,y, game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    # the next direction to be farther from the bomb
    if action == 'UP':
        #agent directly on bomb
        if x == agent_coord_x and y == agent_coord_y:
            #agent on free cross
            if ((agent_coord_x) % 2) != 0 and ((agent_coord_y) % 2) != 0:
                #check if agent is more than 2 fields away from edge
                if agent_coord_y > 2:
                    if game_state['field'][agent_coord_x][agent_coord_y-3] != 0 :
                        if (game_state['field'][agent_coord_x + 1][agent_coord_y-2] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y-2] != 0):
                            return 1
                
                #check if agent is more than 3 fields away from edge
                if agent_coord_y > 3:
                    if ( game_state['field'][agent_coord_x][agent_coord_y-3] != 0 ) or ( game_state['field'][agent_coord_x][agent_coord_y-4] != 0 ):
                        if (game_state['field'][agent_coord_x + 1][agent_coord_y-2] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y-2] != 0):
                            return 1
            
            #agent on |_|_|x|_|_| lane
            if ((agent_coord_y) % 2) == 0 :
                #check if agent is more than 2 fields away from edge
                if agent_coord_y > 2:
                    if game_state['field'][agent_coord_x][agent_coord_y-3] != 0 :
                        if (game_state['field'][agent_coord_x + 1][agent_coord_y-1] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y-1] != 0):
                            return 1
        
        #agent one above bomb
        if x == agent_coord_x and (y-1) == agent_coord_y:
            #agent on free cross
            if ((agent_coord_x) % 2) != 0 and ((agent_coord_y) % 2) != 0:
                #check if agent is more than 2 fields away from edge
                if agent_coord_y > 2:
                    if game_state['field'][agent_coord_x][agent_coord_y-3] != 0 :
                        if (game_state['field'][agent_coord_x + 1][agent_coord_y-2] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y-2] != 0):
                            return 1
            
            #agent on |_|_|x|_|_| lane
            if ((agent_coord_y) % 2) == 0 :
                if (game_state['field'][agent_coord_x + 1][agent_coord_y-1] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y-1] != 0):
                    if agent_coord_y > 1:
                        if game_state['field'][agent_coord_x][agent_coord_y-2] != 0 :
                            return 1
                    if agent_coord_y > 2:
                        if (game_state['field'][agent_coord_x][agent_coord_y-2] != 0) or (game_state['field'][agent_coord_x][agent_coord_y-3] != 0):
                            return 1

        if agent_coord_y == 2:    
            if (x == agent_coord_x and y == agent_coord_y) or (x == agent_coord_x and (y-1) == agent_coord_y):
                if (game_state['field'][agent_coord_x + 1][agent_coord_y-1] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y-1] != 0):
                    return 1 

    


    # the next direction to be farther from the bomb
    if action == 'RIGHT':
        #agent directly on bomb
        if x == agent_coord_x and y == agent_coord_y:
            #agent on free cross
            if ((agent_coord_x) % 2) != 0 and ((agent_coord_y) % 2) != 0:
                #check if agent is more than 2 fields away from edge
                if agent_coord_x < (len(game_state['field'])-1) -2:
                    if game_state['field'][agent_coord_x+3][agent_coord_y] != 0 :
                        if (game_state['field'][agent_coord_x+2][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x+2][agent_coord_y - 1] != 0):
                            return 1
                
                #check if agent is more than 3 fields away from edge
                if agent_coord_x < (len(game_state['field'])-1) -3:
                    if ( game_state['field'][agent_coord_x+3][agent_coord_y] != 0 ) or ( game_state['field'][agent_coord_x+4][agent_coord_y] != 0 ):
                        if (game_state['field'][agent_coord_x+2][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x+2][agent_coord_y - 1] != 0):
                            return 1
            
            #agent on |_|_|x|_|_| lane
            if ((agent_coord_x) % 2) == 0 :
                #check if agent is more than 2 fields away from edge
                if agent_coord_x < (len(game_state['field'])-1) -2:
                    if game_state['field'][agent_coord_x+3][agent_coord_y] != 0 :
                        if (game_state['field'][agent_coord_x+1][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x+1][agent_coord_y - 1] != 0):
                            return 1
        
        #agent one above bomb
        if (x+1) == agent_coord_x and y == agent_coord_y:
            #agent on free cross
            if ((agent_coord_x) % 2) != 0 and ((agent_coord_y) % 2) != 0:
                #check if agent is more than 2 fields away from edge
                if agent_coord_x < (len(game_state['field'])-1) -2:
                    if game_state['field'][agent_coord_x+3][agent_coord_y] != 0 :
                        if (game_state['field'][agent_coord_x+2][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x+2][agent_coord_y - 1] != 0):
                            return 1
            
            #agent on |_|_|x|_|_| lane
            if ((agent_coord_x) % 2) == 0 :
                if (game_state['field'][agent_coord_x+1][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x+1][agent_coord_y - 1] != 0):
                    if agent_coord_x < (len(game_state['field'])-1) -1:
                        if game_state['field'][agent_coord_x+2][agent_coord_y] != 0 :
                            return 1
                    if agent_coord_x < (len(game_state['field'])-1) -2:
                        if (game_state['field'][agent_coord_x+2][agent_coord_y] != 0) or (game_state['field'][agent_coord_x+3][agent_coord_y] != 0):
                            return 1

        if agent_coord_x == (len(game_state['field'])-1) -2:    
            if (x == agent_coord_x and y == agent_coord_y) or ((x+1) == agent_coord_x and y == agent_coord_y):
                if (game_state['field'][agent_coord_x+1][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x+1][agent_coord_y - 1] != 0):
                    return 1 




    # the next direction to be farther from the bomb
    if action == 'DOWN':
        #agent directly on bomb
        if x == agent_coord_x and y == agent_coord_y:
            #agent on free cross
            if ((agent_coord_x) % 2) != 0 and ((agent_coord_y) % 2) != 0:
                #check if agent is more than 2 fields away from edge
                if agent_coord_y < (len(game_state['field'][:][0])-1) -2:
                    if game_state['field'][agent_coord_x][agent_coord_y+3] != 0 :
                        if (game_state['field'][agent_coord_x + 1][agent_coord_y+2] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y+2] != 0):
                            return 1
                
                #check if agent is more than 3 fields away from edge
                if agent_coord_y < (len(game_state['field'][:][0])-1) -3:
                    if ( game_state['field'][agent_coord_x][agent_coord_y+3] != 0 ) or ( game_state['field'][agent_coord_x][agent_coord_y+4] != 0 ):
                        if (game_state['field'][agent_coord_x + 1][agent_coord_y+2] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y+2] != 0):
                            return 1
            
            #agent on |_|_|x|_|_| lane
            if ((agent_coord_y) % 2) == 0 :
                #check if agent is more than 2 fields away from edge
                if agent_coord_y < (len(game_state['field'][:][0])-1) -2:
                    if game_state['field'][agent_coord_x][agent_coord_y+3] != 0 :
                        if (game_state['field'][agent_coord_x + 1][agent_coord_y+1] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y+1] != 0):
                            return 1
        
        #agent one below bomb
        if x == agent_coord_x and (y+1) == agent_coord_y:
            #agent on free cross
            if ((agent_coord_x) % 2) != 0 and ((agent_coord_y) % 2) != 0:
                #check if agent is more than 2 fields away from edge
                if agent_coord_y < (len(game_state['field'][:][0])-1) -2:
                    if game_state['field'][agent_coord_x][agent_coord_y+3] != 0 :
                        if (game_state['field'][agent_coord_x + 1][agent_coord_y+2] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y+2] != 0):
                            return 1
            
            #agent on |_|_|x|_|_| lane
            if ((agent_coord_y) % 2) == 0 :
                if (game_state['field'][agent_coord_x + 1][agent_coord_y+1] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y+1] != 0):
                    if agent_coord_y < (len(game_state['field'][:][0])-1) -1:
                        if game_state['field'][agent_coord_x][agent_coord_y+2] != 0 :
                            return 1
                    if agent_coord_y < (len(game_state['field'][:][0])-1) -2:
                        if (game_state['field'][agent_coord_x][agent_coord_y+2] != 0) or (game_state['field'][agent_coord_x][agent_coord_y+3] != 0):
                            return 1

        if agent_coord_y == (len(game_state['field'][:][0])-1) -2:    
            if (x == agent_coord_x and y == agent_coord_y) or (x == agent_coord_x and (y+1) == agent_coord_y):
                if (game_state['field'][agent_coord_x + 1][agent_coord_y+1] != 0) and (game_state['field'][agent_coord_x - 1][agent_coord_y+1] != 0):
                    return 1 





    # the next direction to be farther from the bomb
    if action == 'LEFT':
        #agent directly on bomb
        if x == agent_coord_x and y == agent_coord_y:
            #agent on free cross
            if ((agent_coord_x) % 2) != 0 and ((agent_coord_y) % 2) != 0:
                #check if agent is more than 2 fields away from edge
                if agent_coord_x > 2:
                    if game_state['field'][agent_coord_x-3][agent_coord_y] != 0 :
                        if (game_state['field'][agent_coord_x-2][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x-2][agent_coord_y - 1] != 0):
                            return 1
                
                #check if agent is more than 3 fields away from edge
                if agent_coord_x > 3:
                    if ( game_state['field'][agent_coord_x-3][agent_coord_y] != 0 ) or ( game_state['field'][agent_coord_x-4][agent_coord_y] != 0 ):
                        if (game_state['field'][agent_coord_x-2][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x-2][agent_coord_y - 1] != 0):
                            return 1
            
            #agent on |_|_|x|_|_| lane
            if ((agent_coord_x) % 2) == 0 :
                #check if agent is more than 2 fields away from edge
                if agent_coord_x > 2:
                    if game_state['field'][agent_coord_x-3][agent_coord_y] != 0 :
                        if (game_state['field'][agent_coord_x-1][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x-1][agent_coord_y - 1] != 0):
                            return 1
        
        #agent one above bomb
        if (x-1) == agent_coord_x and y == agent_coord_y:
            #agent on free cross
            if ((agent_coord_x) % 2) != 0 and ((agent_coord_y) % 2) != 0:
                #check if agent is more than 2 fields away from edge
                if agent_coord_x > 2:
                    if game_state['field'][agent_coord_x-3][agent_coord_y] != 0 :
                        if (game_state['field'][agent_coord_x-2][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x-2][agent_coord_y - 1] != 0):
                            return 1
            
            #agent on |_|_|x|_|_| lane
            if ((agent_coord_x) % 2) == 0 :
                if (game_state['field'][agent_coord_x-1][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x-1][agent_coord_y - 1] != 0):
                    if agent_coord_x > 1:
                        if game_state['field'][agent_coord_x-2][agent_coord_y] != 0 :
                            return 1
                    if agent_coord_x > 2:
                        if (game_state['field'][agent_coord_x-2][agent_coord_y] != 0) or (game_state['field'][agent_coord_x-3][agent_coord_y] != 0):
                            return 1

        if agent_coord_x == 2:    
            if (x == agent_coord_x and y == agent_coord_y) or ((x-1) == agent_coord_x and y == agent_coord_y):
                if (game_state['field'][agent_coord_x-1][agent_coord_y + 1] != 0) and (game_state['field'][agent_coord_x-1][agent_coord_y - 1] != 0):
                    return 1 



    return 0


#done (added)
def move_into_advanced_crate_trap(dangerous_bombs_and_number_of_dangerous_bombs, game_state, action, self):

    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]




    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]

        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y 


        if move_into_advanced_crate_trap_with_this_bomb(x,y, game_state, action, self) == 1:
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


        if move_into_advanced_crate_trap_with_this_bomb(x1,y1, game_state, action, self) == 1:
            return 1
        
        if move_into_advanced_crate_trap_with_this_bomb(x2,y2, game_state, action, self) == 1:
            return 1



    return 0





#done (fixed x and y coords)
def runs_into_explosion(game_state, action, self):

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]


    # the next direction to be farther from the bomb
    if action == 'UP':
        if game_state['explosion_map'][agent_coord_x][agent_coord_y-1] != 0 :
            return 1

    if action == 'RIGHT':
        if game_state['explosion_map'][agent_coord_x+1][agent_coord_y] != 0 :
            return 1

    if action == 'DOWN':
        if game_state['explosion_map'][agent_coord_x][agent_coord_y+1] != 0 :
            return 1

    if action == 'LEFT':
        if game_state['explosion_map'][agent_coord_x-1][agent_coord_y] != 0 :
            return 1


    return 0




#hopefully done (x and y fixed)
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


    # find the 4 closest bombs
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
        now_in_range = [is_bomb_dangerous(i, agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i, agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if not (0 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1


    if action == 'RIGHT':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i, agent_coord_x + 1, agent_coord_y) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i, agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if not (0 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1

    if action == 'DOWN':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i, agent_coord_x, agent_coord_y + 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i, agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if not (0 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1

    if action == 'LEFT':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i, agent_coord_x - 1, agent_coord_y) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i, agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if not (0 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1



    return 0



#hopefully done (x and y fixed)
#exactly the same as 'runs_into_bomb_range_without_dying', but remove the 'not' in the '0 in ...' condition
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


    # find the 4 closest bombs
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
        now_in_range = [is_bomb_dangerous(i, agent_coord_x, agent_coord_y - 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i, agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if (0 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1


    if action == 'RIGHT':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i, agent_coord_x + 1, agent_coord_y) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i, agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if (0 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1

    if action == 'DOWN':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i, agent_coord_x, agent_coord_y + 1) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i, agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if (0 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1

    if action == 'LEFT':
        #compact but shitty to read version of checking if there is at least one bomb into whose range agent would go that he is not currently already in
        now_in_range = [is_bomb_dangerous(i, agent_coord_x - 1, agent_coord_y) for i in closest_bombs ]
        not_already_in_range = [not (is_bomb_dangerous(i, agent_coord_x, agent_coord_y)) for i in closest_bombs ]
        went_into_range = np.logical_and(now_in_range, not_already_in_range)
        condition = np.logical_or(went_into_range, went_into_range)

        if any(condition) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
            #check if you would immediately die and say NOT here (the other function will do this case)
            if (0 in [closest_bombs[i][1] for i in np.where(condition)[0]]):
                return 1



    return 0





#done
def bomb_dropped_although_not_possible(game_state, action, self):

    if action == 'BOMB':
        if ( game_state["self"][2] == False ):
            return 1

    return 0

#done
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
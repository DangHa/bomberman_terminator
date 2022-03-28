from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, runs_into_bomb_range_without_dying, runs_into_explosion
from .callbacks import ACTIONS, FEATURES
from itertools import compress #for 'find_planted_bombs_in_dangerous_range' function

import numpy as np

from .callbacks import ACTION_HISTORY_SIZE, STATE_HISTORY_SIZE


MOVED_INTO_WALL_CRATE = "MOVED_INTO_WALL_CRATE"
# DROPPED_BOMB = "DROPPED_BOMB"
WAITED = "WAITED"
CONTINUED_ACTION_LOOP = "CONTINUED_ACTION_LOOP"
RAN_TOWARDS_CLOSEST_COIN = "RAN_TOWARDS_CLOSEST_COIN"
RAN_AWAY_FROM_CLOSEST_COIN = "RAN_AWAY_FROM_CLOSEST_COIN"
DROPPED_BOMB_IN_RANGE_OF_CRATE = "DROPPED_BOMB_IN_RANGE_OF_CRATE"
RAN_TOWARDS_CLOSEST_CRATE = "RAN_TOWARDS_CLOSEST_CRATE"
RAN_AWAY_FROM_CLOSEST_CRATE = "RAN_AWAY_FROM_CLOSEST_CRATE"
MOVED_ACCORDINGLY_TO_EVENTUALLY_GET_AWAY_FROM_BOMB = "MOVED_ACCORDINGLY_TO_EVENTUALLY_GET_AWAY_FROM_BOMB"
GOT_OUT_OF_BOMB_RANGE = "GOT_OUT_OF_BOMB_RANGE"
GOES_INTO_CRATE_TRAP = "GOES_INTO_CRATE_TRAP"
TRIED_TO_DROP_BOMB_ALTHOUGH_NOT_POSSIBLE = "TRIED_TO_DROP_BOMB_ALTHOUGH_NOT_POSSIBLE"
RAN_INTO_EXPLOSION = "RAN_INTO_EXPLOSION"
RAN_INTO_BOMB_RANGE_WITHOUT_DYING = "RAN_INTO_BOMB_RANGE_WITHOUT_DYING"
RAN_INTO_BOMB_RANGE_WITH_DYING = "RAN_INTO_BOMB_RANGE_WITH_DYING"
GOES_TOWARDS_DANGEROUS_BOMBS = "GOES_TOWARDS_DANGEROUS_BOMBS"
MOVED_INTO_ADVANCED_CRATE_TRAP = "MOVED_INTO_ADVANCED_CRATE_TRAP"


#done
def setup_training(self):
    #hyper_params
    self.epsilon = 0.1 #0.95   EPSILON must be defined in callbacks.py bc in tournament train.py is not called? (do later) 
    self.alpha = 0.2 #0.8
    self.gamma = 0.9 #0.5

    self.logger.info("TRAINING SETUP successful")


#done
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    if self.action_loop_result_before_taken_action != 0:
        self.a = 1
    else:
        self.a = 0

    self.logger.info(f"Value of a: {self.a}")

    if state_to_features(old_game_state, self_action, self) is not None:

        self.former_action.appendleft(0) #CHANGING POINT FOR ACTION LOOP FUNCTION - NORMAL
        event_checker_list = state_to_features(old_game_state, self_action, self)
        self.former_action.append(self.taken_action) #CHANGING POINT FOR ACTION LOOP FUNCTION - FUTURE

        #define events here:

        if event_checker_list[1] != 0:
            events.append(MOVED_INTO_WALL_CRATE)

        if self.a != 0:
            #self.logger.info(f"Added event 'CONTINUED_ACTION_LOOP' with a= {self.a}")
            events.append(CONTINUED_ACTION_LOOP)

        if event_checker_list[3] != 0:
            events.append(RAN_TOWARDS_CLOSEST_COIN)

        if event_checker_list[4] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_COIN)

        if event_checker_list[5] != 0:
            events.append(DROPPED_BOMB_IN_RANGE_OF_CRATE)

        if event_checker_list[6] != 0:
            events.append(RAN_TOWARDS_CLOSEST_CRATE)

        if event_checker_list[7] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_CRATE)

        if event_checker_list[8] != 0:
            events.append(MOVED_ACCORDINGLY_TO_EVENTUALLY_GET_AWAY_FROM_BOMB)

        if event_checker_list[9] != 0:
            events.append(GOT_OUT_OF_BOMB_RANGE)

        if event_checker_list[10] != 0:
            events.append(GOES_INTO_CRATE_TRAP)

        if event_checker_list[11] != 0:
            events.append(TRIED_TO_DROP_BOMB_ALTHOUGH_NOT_POSSIBLE)

        #not useful here bc will never be called bc agent dies when this occures
        # if event_checker_list[12] != 0:
        #     events.append(RAN_INTO_EXPLOSION)

        if event_checker_list[13] != 0:
            events.append(RAN_INTO_BOMB_RANGE_WITHOUT_DYING)

        #not useful here bc will never be called bc agent dies when this occures
        if event_checker_list[14] != 0:
            events.append(RAN_INTO_BOMB_RANGE_WITH_DYING)

        if event_checker_list[15] != 0:
            events.append(GOES_TOWARDS_DANGEROUS_BOMBS)

        if event_checker_list[16] != 0:
            events.append(MOVED_INTO_ADVANCED_CRATE_TRAP)
        

        if event_checker_list[-1] != 0:
            events.append(WAITED)    

        # #only for coin heaven stage
        # if event_checker_list[-2] != 0:
        #     events.append(DROPPED_BOMB)
        # if event_checker_list[-1] != 0:
        #     events.append(WAITED)

        agent_coord_x = old_game_state['self'][3][0]
        agent_coord_y = old_game_state['self'][3][1]

        moved_up = old_game_state['field'][agent_coord_x][agent_coord_y-1] #Up
        moved_ri = old_game_state['field'][agent_coord_x+1][agent_coord_y] #Right
        moved_do = old_game_state['field'][agent_coord_x][agent_coord_y+1] #Down
        moved_le = old_game_state['field'][agent_coord_x-1][agent_coord_y] #Left


        self.logger.info(f'')
        self.logger.info(f'Epsilon: {self.epsilon}')

        self.logger.info(f'Current position:   x: {old_game_state["self"][3][0]}  y: {old_game_state["self"][3][1]}')
        # self.logger.info(f"Field value up: {moved_up}")
        # self.logger.info(f"Field value right: {moved_ri}")
        # self.logger.info(f"Field value down: {moved_do}")
        # self.logger.info(f"Field value left: {moved_le}")
        # self.logger.info(f"Field values: \n {old_game_state['field']}")
        

        #clac R
        R = reward_from_events(self, events)


        #calc Q_max_of_new_s
        weights = self.model
        features_for_action_1 = state_to_features(new_game_state, ACTIONS[0], self)
        features_for_action_2 = state_to_features(new_game_state, ACTIONS[1], self)
        features_for_action_3 = state_to_features(new_game_state, ACTIONS[2], self)
        features_for_action_4 = state_to_features(new_game_state, ACTIONS[3], self)
        features_for_action_5 = state_to_features(new_game_state, ACTIONS[4], self)
        features_for_action_6 = state_to_features(new_game_state, ACTIONS[5], self)

        features_for_all_actions = np.array([
            features_for_action_1,
            features_for_action_2,
            features_for_action_3,
            features_for_action_4,
            features_for_action_5,
            features_for_action_6
        ])

        index_of_best_action = ((weights * features_for_all_actions).sum(axis=1)).argmax(axis=0)
        features_for_best_action = state_to_features(new_game_state, ACTIONS[index_of_best_action], self)

        Q_max_of_new_s = sum(weights * features_for_best_action)



        self.former_action.appendleft(0) #CHANGING POINT FOR ACTION LOOP FUNCTION - NORMAL
        #calc rest for updating
        features = event_checker_list
        self.former_action.append(self.taken_action) #CHANGING POINT FOR ACTION LOOP FUNCTION - FUTURE

        #update weights
        for i in range(FEATURES):
            weights[i] = weights[i] + self.alpha * features[i] * (   R + self.gamma * Q_max_of_new_s - sum(weights * features)   )


        #store weights
        self.model = weights
        #self.logger.info(f"NEW MODEL: \n {self.model}")

        
        self.epsilon = self.epsilon * 0.998
        self.former_state.append(new_game_state)

        
        # self.logger.info(f'Coin map: \n {old_game_state["coins"]}')
        # self.logger.info(f'Coin map type: \n {type(old_game_state["coins"])}')

        


        



#done
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.info(f"-------------------- LAST EVALUATION")

    # #this is true e.g.: 'last_game_state' corresponds to 'old_game_state' in  'game_events_occured' etc.
    old_game_state = last_game_state
    self_action = last_action


    if self.action_loop_result_before_taken_action != 0:
        self.a = 1
    else:
        self.a = 0

    self.logger.info(f"Value of a: {self.a}")

    if state_to_features(old_game_state, self_action, self) is not None:

        self.former_action.appendleft(0) #CHANGING POINT FOR ACTION LOOP FUNCTION - NORMAL
        event_checker_list = state_to_features(old_game_state, self_action, self)
        self.former_action.append(self.taken_action) #CHANGING POINT FOR ACTION LOOP FUNCTION - FUTURE

        #define events here:

        if event_checker_list[1] != 0:
            events.append(MOVED_INTO_WALL_CRATE)

        if self.a != 0:
            #self.logger.info(f"Added event 'CONTINUED_ACTION_LOOP' with a= {self.a}")
            events.append(CONTINUED_ACTION_LOOP)

        if event_checker_list[3] != 0:
            events.append(RAN_TOWARDS_CLOSEST_COIN)

        if event_checker_list[4] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_COIN)

        if event_checker_list[5] != 0:
            events.append(DROPPED_BOMB_IN_RANGE_OF_CRATE)

        if event_checker_list[6] != 0:
            events.append(RAN_TOWARDS_CLOSEST_CRATE)

        if event_checker_list[7] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_CRATE)

        if event_checker_list[8] != 0:
            events.append(MOVED_ACCORDINGLY_TO_EVENTUALLY_GET_AWAY_FROM_BOMB)

        if event_checker_list[9] != 0:
            events.append(GOT_OUT_OF_BOMB_RANGE)

        if event_checker_list[10] != 0:
            events.append(GOES_INTO_CRATE_TRAP)

        if event_checker_list[11] != 0:
            events.append(TRIED_TO_DROP_BOMB_ALTHOUGH_NOT_POSSIBLE)

        #WILL be useful here
        if event_checker_list[12] != 0:
            events.append(RAN_INTO_EXPLOSION)

        #will probably never be called here (I think)
        if event_checker_list[13] != 0:
            events.append(RAN_INTO_BOMB_RANGE_WITHOUT_DYING)

        #WILL be useful here
        if event_checker_list[14] != 0:
            events.append(RAN_INTO_BOMB_RANGE_WITH_DYING)


        if event_checker_list[15] != 0:
            events.append(GOES_TOWARDS_DANGEROUS_BOMBS)

        if event_checker_list[15] != 0:
            events.append(MOVED_INTO_ADVANCED_CRATE_TRAP)
        
        

        if event_checker_list[-1] != 0:
            events.append(WAITED)    

        # #only for coin heaven stage
        # if event_checker_list[-2] != 0:
        #     events.append(DROPPED_BOMB)
        # if event_checker_list[-1] != 0:
        #     events.append(WAITED)

        agent_coord_x = old_game_state['self'][3][0]
        agent_coord_y = old_game_state['self'][3][1]

        moved_up = old_game_state['field'][agent_coord_x][agent_coord_y-1] #Up
        moved_ri = old_game_state['field'][agent_coord_x+1][agent_coord_y] #Right
        moved_do = old_game_state['field'][agent_coord_x][agent_coord_y+1] #Down
        moved_le = old_game_state['field'][agent_coord_x-1][agent_coord_y] #Left


        self.logger.info(f'')
        self.logger.info(f'Epsilon: {self.epsilon}')

        self.logger.info(f'Current position:   x: {old_game_state["self"][3][0]}  y: {old_game_state["self"][3][1]}')
        # self.logger.info(f"Field value up: {moved_up}")
        # self.logger.info(f"Field value right: {moved_ri}")
        # self.logger.info(f"Field value down: {moved_do}")
        # self.logger.info(f"Field value left: {moved_le}")
        # self.logger.info(f"Field values: \n {old_game_state['field']}")
        

        #clac R
        R = reward_from_events(self, events)


        #calc Q_max_of_new_s
        weights = self.model
        Q_max_of_new_s = 0  #<-------------------- SHOULD BE 0 !!! (he said)



        self.former_action.appendleft(0) #CHANGING POINT FOR ACTION LOOP FUNCTION - NORMAL
        #calc rest for updating
        features = event_checker_list
        self.former_action.append(self.taken_action) #CHANGING POINT FOR ACTION LOOP FUNCTION - FUTURE

        #update weights
        for i in range(FEATURES):
            weights[i] = weights[i] + self.alpha * features[i] * (   R + self.gamma * Q_max_of_new_s - sum(weights * features)   )


        #store weights
        self.model = weights
        #self.logger.info(f"NEW MODEL: \n {self.model}")

        
        self.epsilon = self.epsilon * 0.998
        

        
        # self.logger.info(f'Coin map: \n {old_game_state["coins"]}')
        # self.logger.info(f'Coin map type: \n {type(old_game_state["coins"])}')

        


    self.logger.info(f"Chosen wall/crate run count: {self.count_chosen_wall_crate_run} ")
    self.logger.info(f"Chosen action loop count: {self.count_chosen_action_loop} ")
    self.logger.info(f"Chosen crate trap: {self.count_crate_trap} ")
    self.logger.info(f"Chosen suicide (run into explosion): {self.count_runs_into_explosion} ")
    self.logger.info(f"Chosen almost suicide (run into bomb range no dying): {self.count_runs_into_bomb_range_no_dying} ")
    self.logger.info(f"Chosen suicide (run into bomb range with dying): {self.count_runs_into_bomb_range_with_dying} ")
    self.logger.info(f"Chosen advanced crate trap: {self.count_advanced_crate_trap} ")

    
 
    
    
    print()
    print(f"Chosen wall/crate run count: {self.count_chosen_wall_crate_run}")
    print(f"Chosen action loop count: {self.count_chosen_action_loop}")
    print(f"Chosen crate trap count: {self.count_crate_trap}")
    print(f"Chosen suicide (run into explosion): {self.count_runs_into_explosion}")
    print(f"Chosen almost suicide (run into bomb range no dying): {self.count_runs_into_bomb_range_no_dying} ")
    print(f"Chosen suicide (run into bomb range with dying): {self.count_runs_into_bomb_range_with_dying} ")
    print(f"Chosen advanced crate trap: {self.count_advanced_crate_trap} ")


    

    self.logger.info(f"END OF GAME ---------------- Step: {last_game_state['step']} ")

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


    #delete later (and fix)--------------------------------------------------
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


    #hyper_params
    self.epsilon = 0.1 #0.95   EPSILON must be defined in callbacks.py bc in tournament train.py is not called? (do later) 
    self.alpha = 0.2 #0.8
    self.gamma = 0.9 #0.5
    #--------------------------------------------------------------------------

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,

        e.WAITED: 0,
        e.INVALID_ACTION: 0,
        e.BOMB_DROPPED: 0,

        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 0,

        MOVED_INTO_WALL_CRATE: -60, #has to be more penalty than 'continued action loop'
        CONTINUED_ACTION_LOOP: -15,
        RAN_TOWARDS_CLOSEST_COIN: 5,
        RAN_AWAY_FROM_CLOSEST_COIN: -7, #has to be more penalty than 'RAN_TOWARDS_CLOSEST_COIN' has reward, to avoid loop (? unsure)
        #                                 #but not so much penalty bc sometimes agent needs to go around wall

        # #what values should the rewards of coins vs crates have???
        DROPPED_BOMB_IN_RANGE_OF_CRATE: 20, #higher than running towards crate
        RAN_TOWARDS_CLOSEST_CRATE: 7, #higher than running towards coin ???? (for 2nd stage only)
        RAN_AWAY_FROM_CLOSEST_CRATE: -11, #hihger than reward to avoid loop (? unsure) 

        MOVED_ACCORDINGLY_TO_EVENTUALLY_GET_AWAY_FROM_BOMB: 30, #higher than running towards coin or crate and higher than their sum
                                                                #hihger than reward for 'planting bomb' action of any kind
                                                                #higher than the punishment of moving away from crate (coin doesnt matter)
        GOT_OUT_OF_BOMB_RANGE: 45,  #higher than GET_AWAY_FROM_BOMB (how much ?)
        GOES_INTO_CRATE_TRAP: -35,  #higher than award for running towards crate or coin (higher than their sum)  
                                    #higher than running away from crate   
                                    #higher than reward for moving away from bomb (higher than sum of all)                                                 
        TRIED_TO_DROP_BOMB_ALTHOUGH_NOT_POSSIBLE: -20, #higher than running away from coin but not higher than 'MOVED_INTO_WALL_CRATE'
        
        RAN_INTO_EXPLOSION: -60, #VERY HIGH (this is equivalent to dying)
        RAN_INTO_BOMB_RANGE_WITHOUT_DYING: -50, #higher than reward for moving out of bomb range
        RAN_INTO_BOMB_RANGE_WITH_DYING: -60, #same as 'RAN_INTO_EXPLOSION' --> instant death

        GOES_TOWARDS_DANGEROUS_BOMBS: -15, #higher than running towards coin or crate and higher than their sum
        MOVED_INTO_ADVANCED_CRATE_TRAP: -35, #maybe the same as moving into std crate trap (could be a tiny bit worse)

        WAITED: -2, #less punishment than running into bomb



        #CAUTION! escaping from bomb has to give greater  


        # #only needed for coin heaven stage
        # DROPPED_BOMB: -20,
        # WAITED: -5,

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    
    #for logger
    if self.random_or_choosen == 1:
        self.logger.info(f"RANDOM ACTION: {self.former_action[-1]}")
    if self.random_or_choosen == 2:
        self.logger.info(f"CHOSEN ACTION: {self.former_action[-1]}")


    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    #for logger
    if MOVED_INTO_WALL_CRATE in events and self.random_or_choosen == 2:
        self.count_chosen_wall_crate_run += 1
    #for logger
    if CONTINUED_ACTION_LOOP in events and self.random_or_choosen == 2:
        self.count_chosen_action_loop += 1
    #for logger
    if GOES_INTO_CRATE_TRAP in events and self.random_or_choosen == 2:
        self.count_crate_trap += 1
    #for logger
    if RAN_INTO_EXPLOSION in events and self.random_or_choosen == 2:
        self.count_runs_into_explosion += 1
    #for logger
    if RAN_INTO_BOMB_RANGE_WITHOUT_DYING in events and self.random_or_choosen == 2:
        self.count_runs_into_bomb_range_no_dying += 1
    #for logger
    if RAN_INTO_BOMB_RANGE_WITH_DYING in events and self.random_or_choosen == 2:
        self.count_runs_into_bomb_range_with_dying += 1
    #for logger
    if MOVED_INTO_ADVANCED_CRATE_TRAP in events and self.random_or_choosen == 2:
        self.count_advanced_crate_trap += 1
    

    return reward_sum

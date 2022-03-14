from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, runs_into_bomb_range_without_dying, runs_into_explosion
from .callbacks import ACTIONS, FEATURES

import numpy as np


MOVED_INTO_WALL_CRATE = "MOVED_INTO_WALL_CRATE"
DROPPED_BOMB = "DROPPED_BOMB"
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


        #define events here:

        if state_to_features(old_game_state, self_action, self)[1] != 0:
            events.append(MOVED_INTO_WALL_CRATE)

        if self.a != 0:
            #self.logger.info(f"Added event 'CONTINUED_ACTION_LOOP' with a= {self.a}")
            events.append(CONTINUED_ACTION_LOOP)

        if state_to_features(old_game_state, self_action, self)[3] != 0:
            events.append(RAN_TOWARDS_CLOSEST_COIN)

        if state_to_features(old_game_state, self_action, self)[4] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_COIN)

        if state_to_features(old_game_state, self_action, self)[5] != 0:
            events.append(DROPPED_BOMB_IN_RANGE_OF_CRATE)

        if state_to_features(old_game_state, self_action, self)[6] != 0:
            events.append(RAN_TOWARDS_CLOSEST_CRATE)

        if state_to_features(old_game_state, self_action, self)[7] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_CRATE)

        if state_to_features(old_game_state, self_action, self)[8] != 0:
            events.append(MOVED_ACCORDINGLY_TO_EVENTUALLY_GET_AWAY_FROM_BOMB)

        if state_to_features(old_game_state, self_action, self)[9] != 0:
            events.append(GOT_OUT_OF_BOMB_RANGE)

        if state_to_features(old_game_state, self_action, self)[10] != 0:
            events.append(GOES_INTO_CRATE_TRAP)

        if state_to_features(old_game_state, self_action, self)[11] != 0:
            events.append(TRIED_TO_DROP_BOMB_ALTHOUGH_NOT_POSSIBLE)

        if state_to_features(old_game_state, self_action, self)[12] != 0:
            events.append(RAN_INTO_EXPLOSION)

        if state_to_features(old_game_state, self_action, self)[13] != 0:
            events.append(RAN_INTO_BOMB_RANGE_WITHOUT_DYING)

        if state_to_features(old_game_state, self_action, self)[14] != 0:
            events.append(RAN_INTO_BOMB_RANGE_WITH_DYING)

        if state_to_features(old_game_state, self_action, self)[-1] != 0:
            events.append(WAITED)    

        # #only for coin heaven stage
        # if state_to_features(old_game_state, self_action, self)[-2] != 0:
        #     events.append(DROPPED_BOMB)
        # if state_to_features(old_game_state, self_action, self)[-1] != 0:
        #     events.append(WAITED)

        agent_coord_x = old_game_state['self'][3][0]
        agent_coord_y = old_game_state['self'][3][1]

        moved_up = old_game_state['field'][agent_coord_x][agent_coord_y-1] #Up
        moved_ri = old_game_state['field'][agent_coord_x+1][agent_coord_y] #Right
        moved_do = old_game_state['field'][agent_coord_x][agent_coord_y+1] #Down
        moved_le = old_game_state['field'][agent_coord_x-1][agent_coord_y] #Left

        self.logger.info(f"Field value up: {moved_up}")
        self.logger.info(f"Field value right: {moved_ri}")
        self.logger.info(f"Field value down: {moved_do}")
        self.logger.info(f"Field value left: {moved_le}")
        self.logger.info(f"Field values: \n {old_game_state['field']}")

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


        #calc rest for updating
        features = state_to_features(old_game_state, self_action, self)

        #update weights
        for i in range(FEATURES):
            weights[i] = weights[i] + self.alpha * features[i] * (   R + self.gamma * Q_max_of_new_s - sum(weights * features)   )


        #store weights
        self.model = weights
        self.logger.info(f"NEW MODEL: \n {self.model}")

        
        self.epsilon = self.epsilon * 0.998
        self.former_state.append(new_game_state)

        self.logger.info(f'Current position:   x: {old_game_state["self"][3][0]}  y: {old_game_state["self"][3][1]}')
        # self.logger.info(f'Coin map: \n {old_game_state["coins"]}')
        # self.logger.info(f'Coin map type: \n {type(old_game_state["coins"])}')

        self.logger.info(f'------------------------------------ Step: {new_game_state["step"]}')


        



#done
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    #this is true e.g.: 'last_game_state' corresponds to 'old_game_state' in  'game_events_occured' etc.
    old_game_state = last_game_state
    self_action = last_action



    #same as 'game_events_occured'-----------------------------------


    if self.action_loop_result_before_taken_action != 0:
        self.a = 1
    else:
        self.a = 0

    self.logger.info(f"Value of a: {self.a}")

    if state_to_features(old_game_state, self_action, self) is not None:


        #define events here:

        if state_to_features(old_game_state, self_action, self)[1] != 0:
            events.append(MOVED_INTO_WALL_CRATE)

        if self.a != 0:
            #self.logger.info(f"Added event 'CONTINUED_ACTION_LOOP' with a= {self.a}")
            events.append(CONTINUED_ACTION_LOOP)

        if state_to_features(old_game_state, self_action, self)[3] != 0:
            events.append(RAN_TOWARDS_CLOSEST_COIN)

        if state_to_features(old_game_state, self_action, self)[4] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_COIN)

        if state_to_features(old_game_state, self_action, self)[5] != 0:
            events.append(DROPPED_BOMB_IN_RANGE_OF_CRATE)

        if state_to_features(old_game_state, self_action, self)[6] != 0:
            events.append(RAN_TOWARDS_CLOSEST_CRATE)

        if state_to_features(old_game_state, self_action, self)[7] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_CRATE)

        if state_to_features(old_game_state, self_action, self)[8] != 0:
            events.append(MOVED_ACCORDINGLY_TO_EVENTUALLY_GET_AWAY_FROM_BOMB)

        if state_to_features(old_game_state, self_action, self)[9] != 0:
            events.append(GOT_OUT_OF_BOMB_RANGE)

        if state_to_features(old_game_state, self_action, self)[10] != 0:
            events.append(GOES_INTO_CRATE_TRAP)

        if state_to_features(old_game_state, self_action, self)[11] != 0:
            events.append(TRIED_TO_DROP_BOMB_ALTHOUGH_NOT_POSSIBLE)

        if state_to_features(old_game_state, self_action, self)[12] != 0:
            events.append(RAN_INTO_EXPLOSION)

        if state_to_features(old_game_state, self_action, self)[13] != 0:
            events.append(RAN_INTO_BOMB_RANGE_WITHOUT_DYING)

        if state_to_features(old_game_state, self_action, self)[14] != 0:
            events.append(RAN_INTO_BOMB_RANGE_WITH_DYING)

        if state_to_features(old_game_state, self_action, self)[-1] != 0:
            events.append(WAITED)    

        # #only for coin heaven stage
        # if state_to_features(old_game_state, self_action, self)[-2] != 0:
        #     events.append(DROPPED_BOMB)
        # if state_to_features(old_game_state, self_action, self)[-1] != 0:
        #     events.append(WAITED)

        agent_coord_x = old_game_state['self'][3][0]
        agent_coord_y = old_game_state['self'][3][1]

        moved_up = old_game_state['field'][agent_coord_x][agent_coord_y-1] #Up
        moved_ri = old_game_state['field'][agent_coord_x+1][agent_coord_y] #Right
        moved_do = old_game_state['field'][agent_coord_x][agent_coord_y+1] #Down
        moved_le = old_game_state['field'][agent_coord_x-1][agent_coord_y] #Left

        self.logger.info(f"Field value up: {moved_up}")
        self.logger.info(f"Field value right: {moved_ri}")
        self.logger.info(f"Field value down: {moved_do}")
        self.logger.info(f"Field value left: {moved_le}")
        self.logger.info(f"Field values: \n {old_game_state['field']}")

        #clac R
        R = reward_from_events(self, events)



        #slightly different than 'game_events_occured'-----------------------

        
        #get weights
        weights = self.model

        #calc rest for updating
        features = state_to_features(old_game_state, last_action, self)

        #calc Q_max_of_new_s
        Q_max_of_new_s = 0 #<-------------------- SHOULD BE 0 !!! (he said)

        #update weights
        for i in range(FEATURES):
            weights[i] = weights[i] + self.alpha * features[i] * (   R + self.gamma * Q_max_of_new_s - sum(weights * features)   )


        #store weights
        self.model = weights
        self.logger.info(f"NEW MODEL: \n {self.model}")

        
        self.epsilon = self.epsilon * 0.998


        self.logger.info(f'Current position:   x: {last_game_state["self"][3][0]}  y: {last_game_state["self"][3][1]}')
        # self.logger.info(f'Coin map: \n {old_game_state["coins"]}')
        # self.logger.info(f'Coin map type: \n {type(old_game_state["coins"])}')

        self.logger.info(f"Test stored last state, former coordinates: {self.former_state[0]['self'][3]}")  

        self.logger.info(f"Run into explosion UP: {runs_into_explosion(old_game_state, ACTIONS[0], self)}") 
        self.logger.info(f"Run into explosion RIGHT: {runs_into_explosion(old_game_state, ACTIONS[1], self)}") 
        self.logger.info(f"Run into explosion DOWN: {runs_into_explosion(old_game_state, ACTIONS[2], self)}") 
        self.logger.info(f"Run into explosion LEFT: {runs_into_explosion(old_game_state, ACTIONS[3], self)}") 

        self.logger.info(f"runs_into_bomb_range_without_dying: {runs_into_bomb_range_without_dying(last_game_state, last_action, self)}")
        self.logger.info(f"runs_into_bomb_range_with_dying: {runs_into_bomb_range_without_dying(last_game_state, last_action, self)}")


    self.logger.info(f"END OF GAME ---------------- Step: {last_game_state['step']} ")

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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

        MOVED_INTO_WALL_CRATE: -50, #has to be more penalty than 'continued action loop'
        CONTINUED_ACTION_LOOP: -15,
        RAN_TOWARDS_CLOSEST_COIN: 5,
        RAN_AWAY_FROM_CLOSEST_COIN: -7, #has to be more penalty than 'RAN_TOWARDS_CLOSEST_COIN' has reward, to avoid loop (? unsure)
                                        #but not so much penalty bc sometimes agent needs to go around wall

        #what values should the rewards of coins vs crates have???
        DROPPED_BOMB_IN_RANGE_OF_CRATE: 20, #higher than running towards crate
        RAN_TOWARDS_CLOSEST_CRATE: 7, #higher than running towards coin ???? (for 2nd stage only)
        RAN_AWAY_FROM_CLOSEST_CRATE: -11, #hihger than reward to avoid loop (? unsure) 

        MOVED_ACCORDINGLY_TO_EVENTUALLY_GET_AWAY_FROM_BOMB: 30, #higher than running towards coin or crate and higher than their sum
                                                                #hihger than reward than 'planting bomb' action of any kind
                                                                #higher than the punishment of moving away from crate (coin doesnt matter)
        GOT_OUT_OF_BOMB_RANGE: 40, #higher than GET_AWAY_FROM_BOMB (how much ?)
        GOES_INTO_CRATE_TRAP: -50,  #higher than award for running towards crate or coin (higher than their sum)  
                                    #higher than running away from crate   
                                    # higher than reward for moving away from bomb (higher than sum of all)                                                 
        TRIED_TO_DROP_BOMB_ALTHOUGH_NOT_POSSIBLE: -10, #higher than running away from coin but not higher than 'MOVED_INTO_WALL_CRATE'
        RAN_INTO_EXPLOSION: -50, #VERY HIGH (this is equivalent to dying)
        RAN_INTO_BOMB_RANGE_WITHOUT_DYING: -40,
        RAN_INTO_BOMB_RANGE_WITHOUT_DYING: -50, #same as 'RAN_INTO_EXPLOSION' --> instant death

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
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
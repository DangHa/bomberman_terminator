from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import get_state_index_from_game_state, runs_into_wall_crate, closest_coin_but_not_wall_or_crate
from .callbacks import ACTIONS, FEATURES

import numpy as np


MOVED_INTO_WALL_CRATE = "MOVED_INTO_WALL_CRATE"
RAN_TOWARDS_CLOSEST_COIN = "RAN_TOWARDS_CLOSEST_COIN"
DID_NOT_RUN_TOWARDS_CLOSEST_COIN = "DID_NOT_RUN_TOWARDS_CLOSEST_COIN"


#done
def setup_training(self):
    #hyper_params
    self.alpha = 0.2 #0.8
    self.gamma = 0.9 #0.5

    #action to index conversion help
    self.action_index_of = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    self.logger.info("TRAINING SETUP successful")


#done
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    if old_game_state is not None:
    
        
        #define events here:

        if runs_into_wall_crate(old_game_state, self_action, self) == 1:
            events.append(MOVED_INTO_WALL_CRATE)

        if closest_coin_but_not_wall_or_crate(old_game_state, self) == (self.action_index_of[self_action] + 1):
            events.append(RAN_TOWARDS_CLOSEST_COIN)
        else:
            events.append(DID_NOT_RUN_TOWARDS_CLOSEST_COIN)


        #get stuff for updating formula
        R = reward_from_events(self, events)
        state_index = get_state_index_from_game_state(old_game_state, self)
        q_value = self.model[state_index][self.action_index_of[self_action]]
        
        new_state_index = get_state_index_from_game_state(new_game_state, self) 
        max_new_q_value = np.amax(self.model[new_state_index])

        #updating formula for q_table entry
        q_value = q_value + self.alpha * (R + self.gamma * max_new_q_value - q_value )

        #update this q_table entry
        self.model[state_index][self.action_index_of[self_action]] = q_value


        #reduce epsilon
        self.epsilon = self.epsilon * 0.998

        # #for checking coin maps
        # self.logger.info(f'Coin map: \n {old_game_state["coins"]}')
        # self.logger.info(f'Coin map type: \n {type(old_game_state["coins"])}')

        


        



#done
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    
    

    self.logger.info(f"END OF GAME ---------------- Step: {last_game_state['step']} ")

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


    #due to shitty flag command '--n-rounds' have to redo necessary setup after each match here 

    #hyper_params
    self.epsilon = 0.10 #0.95   EPSILON must be defined in callbacks.py bc in tournament train.py is not called? (do later) 
    self.alpha = 0.2 #0.8
    self.gamma = 0.9 #0.5
    #--------------------------------------------------------------------------

def reward_from_events(self, events: List[str]) -> int:

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

        MOVED_INTO_WALL_CRATE: -50, #has to be high penalty
        RAN_TOWARDS_CLOSEST_COIN: 5, #has to be relatively small reward
        DID_NOT_RUN_TOWARDS_CLOSEST_COIN: -6 #has to be higher than reward for moving towards coin, 
        #                                    #but lower than penalty for running into wall
        
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]


    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    

    return reward_sum

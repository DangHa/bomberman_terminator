from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS, FEATURES

import numpy as np


MOVED_INTO_WALL_CRATE = "MOVED_INTO_WALL_CRATE"
DROPPED_BOMB = "DROPPED_BOMB"
WAITED = "WAITED"
CONTINUED_ACTION_LOOP = "CONTINUED_ACTION_LOOP"
RAN_TOWARDS_CLOSEST_COIN = "RAN_TOWARDS_CLOSEST_COIN"
RAN_AWAY_FROM_CLOSEST_COIN = "RAN_AWAY_FROM_CLOSEST_COIN"

#done
def setup_training(self):
    #hyper_params
    self.epsilon = 0.95 #0.1
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


        if state_to_features(old_game_state, self_action, self)[2] != 0:
            events.append(DROPPED_BOMB)

        if state_to_features(old_game_state, self_action, self)[3] != 0:
            events.append(WAITED)

        if self.a != 0:
            self.logger.info(f"Added event 'CONTINUED_ACTION_LOOP' with a= {self.a}")
            events.append(CONTINUED_ACTION_LOOP)

        if state_to_features(old_game_state, self_action, self)[5] != 0:
            events.append(RAN_TOWARDS_CLOSEST_COIN)

        if state_to_features(old_game_state, self_action, self)[6] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_COIN)



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

        
        self.epsilon = self.epsilon * 0.98


        self.logger.info(f'Current position: \n x: {old_game_state["self"][3][0]}  y: {old_game_state["self"][3][1]}')
        self.logger.info(f'Coin map: \n {old_game_state["coins"]}')

        self.logger.info(f'------------------------------------ Step: {new_game_state["step"]}')


        



#done
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.info(f"END OF GAME")

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

        MOVED_INTO_WALL_CRATE: -5,
        DROPPED_BOMB: -20,
        WAITED: -5,
        CONTINUED_ACTION_LOOP: -15,
        RAN_TOWARDS_CLOSEST_COIN: 5,
        RAN_AWAY_FROM_CLOSEST_COIN: -7, #has to be more penalty than 'RAN_TOWARDS_CLOSEST_COIN' has reward to avoid loop

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

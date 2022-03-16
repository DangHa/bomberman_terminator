from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import ACTIONS, FEATURES, act, state_to_features, feature_for_all_action

import numpy as np

# Events
MOVED_INTO_WALL_CRATE = "MOVED_INTO_WALL_CRATE"
DROPPED_BOMB = "DROPPED_BOMB"
WAITED = "WAITED"
CONTINUED_ACTION_LOOP = "CONTINUED_ACTION_LOOP"
RAN_TOWARDS_CLOSEST_COIN = "RAN_TOWARDS_CLOSEST_COIN"
RAN_AWAY_FROM_CLOSEST_COIN = "RAN_AWAY_FROM_CLOSEST_COIN"
DROPPED_BOMB_IN_RANGE_OF_CRATE = "DROPPED_BOMB_IN_RANGE_OF_CRATE"
RAN_TOWARDS_CLOSEST_CRATE = "RAN_TOWARDS_CLOSEST_CRATE"



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    #hyper_params
    self.epsilon = 0.1 #0.95   EPSILON must be defined in callbacks.py bc in tournament train.py is not called? (do later) 
    self.alpha = 0.2 #0.8
    self.gamma = 0.9 #0.5

    self.logger.info("TRAINING SETUP successful")

    self.batch_size = 10
    self.batch_content = np.zeros(len(ACTIONS), dtype=int)
    self.s_batch = np.zeros([len(ACTIONS), self.batch_size, FEATURES])
    self.y_batch = np.zeros([len(ACTIONS), self.batch_size])
    self.gamma = 0.05
    self.logger.info("Training setup.")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.logger.debug("Attempted action: {}".format(self_action))

    if state_to_features(old_game_state, self_action, self) is not None:

        if self.action_loop_result_before_taken_action != 0:
            self.a = 1
        else:
            self.a = 0

        self.logger.info(f"Value of a: {self.a}")

        #define events here:
        if state_to_features(old_game_state, self_action, self)[1] != 0:
            events.append(MOVED_INTO_WALL_CRATE)

        if self.a != 0:
            self.logger.info(f"Added event 'CONTINUED_ACTION_LOOP' with a= {self.a}")
            events.append(CONTINUED_ACTION_LOOP)

        if state_to_features(old_game_state, self_action, self)[3] != 0:
            events.append(RAN_TOWARDS_CLOSEST_COIN)

        if state_to_features(old_game_state, self_action, self)[4] != 0:
            events.append(RAN_AWAY_FROM_CLOSEST_COIN)

        if state_to_features(old_game_state, self_action, self)[5] != 0:
            events.append(DROPPED_BOMB_IN_RANGE_OF_CRATE)

        if state_to_features(old_game_state, self_action, self)[6] != 0:
            events.append(RAN_TOWARDS_CLOSEST_CRATE)

        #only for coin heaven stage
        if state_to_features(old_game_state, self_action, self)[-2] != 0:
            events.append(DROPPED_BOMB)
        if state_to_features(old_game_state, self_action, self)[-1] != 0:
            events.append(WAITED)

        print("EVENT: ", events)

        #clac R
        R = reward_from_events(self, events)

        add_to_batch(self, old_game_state, self_action, new_game_state,events, R)

    train_forest(self)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    #Train Trees even if batch is not full yet
    for i in range(len(self.batch_content)):
        if self.batch_content[i]>0:
            self.forests[i].n_estimators += 1
            self.forests[i].fit(self.s_batch[i][:self.batch_content[i]],self.y_batch[i][:self.batch_content[i]])
            self.logger.debug("Forest {} was trained.".format(i))
            self.s_batch[i] = np.zeros([self.batch_size, FEATURES])
            self.y_batch[i] = np.zeros(self.batch_size)
            self.batch_content[i] = 0


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.forests, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    #MODIFY THE FOLLOWING REWARDS
    game_rewards = {
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,

        e.WAITED: -5,
        e.INVALID_ACTION: 0,
        e.BOMB_DROPPED: 0,

        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 5,

        MOVED_INTO_WALL_CRATE: -50, #has to be more penalty than 'continued action loop'
        CONTINUED_ACTION_LOOP: -15,
        RAN_TOWARDS_CLOSEST_COIN: 5,
        RAN_AWAY_FROM_CLOSEST_COIN: -7, #has to be more penalty than 'RAN_TOWARDS_CLOSEST_COIN' has reward, to avoid loop (? unsure)
                                        #but not so much penalty bc sometimes agent needs to go around wall

        #what values should the rewards of coins vs crates have???
        DROPPED_BOMB_IN_RANGE_OF_CRATE: 30, #higher than running towards crate
        RAN_TOWARDS_CLOSEST_CRATE:  10, #higher than running towards coin ???? (for 2nd stage only)

        # #only needed for coin heaven stage
        DROPPED_BOMB: -20,
        WAITED: -5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def action_to_index(action: str):
    return ACTIONS.index(action)

def y_function(self, R, new_game_state):
    features_for_all_actions = feature_for_all_action(new_game_state, self)

    Y = np.array([self.forests[i].predict(features_for_all_actions)[0] for i in range(len(ACTIONS))])
    self.logger.debug("Training: New target value Y was calculated to {}".format(R+self.gamma*np.max(Y)))
    return R+self.gamma*np.max(Y)

def add_to_batch(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str], R):
    action_index = action_to_index(self_action)
    s = state_to_features(old_game_state, ACTIONS[action_index], self)
    Y = y_function(self, R, new_game_state) 

    self.s_batch[action_index][self.batch_content[action_index]] = s
    self.y_batch[action_index][self.batch_content[action_index]] = Y
    self.batch_content[action_index] += 1

    self.logger.debug("New state s and target Y added to batch")

def train_forest(self):
    if np.any(self.batch_content == self.batch_size):
        action_index = np.where(self.batch_content==self.batch_size)[0][0]
        self.logger.debug("Maximum batch size reached for action {}".format(action_index))
        self.forests[action_index].n_estimators += 1
        self.forests[action_index].fit(self.s_batch[action_index], self.y_batch[action_index])
        self.logger.debug("Forest {} was trained.".format(action_index))
        self.s_batch[action_index] = np.zeros([self.batch_size, FEATURES])
        self.y_batch[action_index] = np.zeros(self.batch_size)
        self.batch_content[action_index] = 0
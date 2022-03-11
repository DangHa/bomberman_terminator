from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import ACTIONS, FEATURES, act, state_to_features

import numpy as np

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.batch_size = 10
    self.batch_content = np.zeros(len(ACTIONS), dtype=int)
    self.s_batch = np.zeros([len(ACTIONS),self.batch_size,len(FEATURES)])
    self.y_batch = np.zeros([len(ACTIONS),self.batch_size])
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

    if old_game_state is None:
        self.logger.debug("Model was not updated, because old_game_state is None.")

    else:
        add_to_batch(self,old_game_state,self_action,new_game_state,events)

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
        e.MOVED_LEFT: 5,
        e.MOVED_RIGHT: 5,
        e.MOVED_UP: 5,
        e.MOVED_DOWN: 5,

        e.WAITED: -3,
        e.INVALID_ACTION: -5,

        e.BOMB_DROPPED: -10,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED:0,

        e.KILLED_OPPONENT: 0,
        e.OPPONENT_ELIMINATED: 0,

        e.KILLED_SELF: 0,
        e.GOT_KILLED: 0,

        e.SURVIVED_ROUND: 0,

        PLACEHOLDER_EVENT: 0  
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
    s_new = state_to_features(self,new_game_state)
    Y = np.array([self.forests[i].predict(s_new.reshape(1, -1))[0] for i in range(len(ACTIONS))])
    self.logger.debug("Training: New target value Y was calculated to {}".format(R+self.gamma*np.max(Y)))
    return R+self.gamma*np.max(Y)

def add_to_batch(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    action_index = action_to_index(self_action)
    R = reward_from_events(self,events)
    s = state_to_features(self,old_game_state)
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
        self.forests[action_index].fit(self.s_batch[action_index],self.y_batch[action_index])
        self.logger.debug("Forest {} was trained.".format(action_index))
        self.s_batch[action_index] = np.zeros([self.batch_size,len(FEATURES)])
        self.y_batch[action_index] = np.zeros(self.batch_size)
        self.batch_content[action_index] = 0




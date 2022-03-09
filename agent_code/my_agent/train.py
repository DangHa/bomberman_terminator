from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import ACTIONS, FEATURES, q_function, state_to_features

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state','weights', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
CLOSE_COIN_EVENT = "CLOSECOIN" # have to save the closest coin from previous state --> should we do it or not?


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #For now I'm initializing everything to zero
    self.epsilon = 0.1
    self.alpha = 0.8
    self.gamma = 0.5
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
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

    # ToDO: Add your own events to hand out rewards
    """
    if ...:
        events.append(PLACEHOLDER_EVENT)
    """

    action_index = action_to_index(self_action)
    R = reward_from_events(self,events)

    if old_game_state is None:
        new_weight = self.weights
        self.logger.debug("Model was not updated, because old_game_state is None.")
        transition = Transition(state_to_features(self,old_game_state), self.weights, self_action, state_to_features(self,new_game_state),R)

    else:
        s = state_to_features(self,old_game_state)
        q_temp = q_function(self, new_game_state, self.weights)

        new_weight = self.weights + self.alpha * s[:,action_index] * (R + self.gamma*np.max(q_temp) - self.q_values[action_index])
        self.logger.debug("Model succesfully updated.")
        transition = Transition(s, self.weights, self_action, state_to_features(self, new_game_state), R)
        print(events)

    self.transitions.append(transition)
    self.weights = new_weight
    


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
    self.transitions.append(Transition(state_to_features(self,last_game_state), self.weights, last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.weights, file)


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
        e.INVALID_ACTION: -10,

        e.BOMB_DROPPED: -10,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 3,
        e.COIN_FOUND: 5,
        e.COIN_COLLECTED:10,

        e.KILLED_OPPONENT: 0,
        e.OPPONENT_ELIMINATED: 0,

        e.KILLED_SELF: 0,
        e.GOT_KILLED: 0,

        e.SURVIVED_ROUND: 10,

        PLACEHOLDER_EVENT: 0  # ADD CUSTOM EVENTS?
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def action_to_index(action: str):
    return ACTIONS.index(action)


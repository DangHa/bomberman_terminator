from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

###___SET UP___: 
"""
If a quantity is changed during learning process it is stored in obj 'self' 
Else it is stored as a global variable here
"""

##__GLOBAL TRAINING VARS__:

# Transition tuple declaration
Transition = namedtuple('Transition', ('s', 'a', 'new_s', 'r'))
action_list = [0]

# Hyper parameters for learning
alpha = 0.1
gamma = 0.1
TRANSITION_HISTORY_SIZE = 2  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
MOVED_OUT_OF_BOMB_RANGE = "MOVED_OUT_OF_BOMB_RANGE"
MOVED_INTO_BOMB_RANGE = "MOVED_INTO_BOMB_RANGE"
GOES_AWAY_FROM_BOMB = "GOES_AWAY_FROM_BOMB"
MOVED_AGAINST_WALL = "MOVED_AGAINST_WALL"
TRIED_BOMB_BUT_NOT_POSSIBLE = "TRIED_BOMB_BUT_NOT_POSSIBLE"

##__STORING IN 'SELF'__
def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    ##__needed fixed values in 'callbacks.py'__

    """No fixed values are needed in both files"""


    ##__mutable values__
    
    #transition tuple (s, a, r, s') stored as deque that holds last ... transitions
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    #set up storage for bomb reward
    self.num = 0

    


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    #calc reward:
    #r = reward_from_events(self, events)


    #store transition in self <------------------------------------------------ (why?)

    # state_to_features is defined in callbacks.py <---------------------------------------- maybe for us not necessary
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    self.logger.debug(f'Successfully stored transition in step {new_game_state["step"]}')

    index = action_to_index(self_action)
    self.logger.debug(f'Successfully calculated action index in step {new_game_state["step"]}, with index {index}')

    if state_to_features(old_game_state) is not None:

        # Idea: Add your own events to hand out rewards

        #___define events___


        self_coord = old_game_state["self"][3]
        self_coord_new = new_game_state["self"][3]
        self.logger.debug(f'Successfully calculated self_coord_old {self_coord}')
        self.logger.debug(f'Successfully calculated self_coord_new {self_coord_new}')
    
        #___avoid bombs___
        # self.logger.debug(f'Explosion map: {old_game_state["explosion_map"]}')
        # self.logger.debug(f'Explosion map: {old_game_state["explosion_map"][self_coord]}')

        # #__was not in bomb range__
        # if old_game_state["explosion_map"][self_coord] == 0:

        # #goes into of bomb range
        #     if old_game_state["explosion_map"][self_coord] < new_game_state["explosion_map"][self_coord]:
        #         events.append(MOVED_INTO_BOMB_RANGE)
        #         self.logger.debug(f'Successfully appended event {MOVED_INTO_BOMB_RANGE}')
        
        # #goes out of bomb range
        #     if old_game_state["explosion_map"][self_coord] > new_game_state["explosion_map"][self_coord]:
        #         events.append(MOVED_OUT_OF_BOMB_RANGE)
        #         self.logger.debug(f'Successfully appended event {MOVED_OUT_OF_BOMB_RANGE}')


        # #__was in bomb range__
        # if old_game_state["explosion_map"][self_coord] != 0:

        #    #goes into of bomb range (+1 bc value reduces each round)
        #     if old_game_state["explosion_map"][self_coord] < new_game_state["explosion_map"][self_coord] + 1:
        #         events.append(MOVED_INTO_BOMB_RANGE)
        #         self.logger.debug(f'Successfully appended event {MOVED_INTO_BOMB_RANGE}')

        #     #goes out of bomb range (+1 bc value reduces each round)
        #     if old_game_state["explosion_map"][self_coord] > new_game_state["explosion_map"][self_coord] + 1:
        #         events.append(MOVED_OUT_OF_BOMB_RANGE)
        #         self.logger.debug(f'Successfully appended event {MOVED_OUT_OF_BOMB_RANGE}')
        
           
        #     bombs_in_range = get_bombs_in_range(self_coord,old_game_state["bombs"][0])

        #     cond, r = goes_away_from_bomb(self_coord, bombs_in_range, self_coord_new)

        #     if cond:
        #         self.num = r
        #         events.append(GOES_AWAY_FROM_BOMB)
        #         self.logger.debug(f'Successfully appended event {GOES_AWAY_FROM_BOMB}')
        


        #goes 1 box away from bomb
        # if old_game_state["explosion_map"][self_coord] != 0:
        # old_game_state["self"][3]
        # old_game_state["bombs"][0]
        #     [self_coord] > new_game_state["explosion_map"][self_coord]

        moved_to_coord = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5, }

        if moved_to_coord[self_action] <= 3:  

            if state_to_features(old_game_state)[moved_to_coord[self_action]] == 1:
                events.append(MOVED_AGAINST_WALL)
                self.logger.debug(f'Successfully appended event {MOVED_AGAINST_WALL}')

        if not old_game_state["self"][2] and self_action == 'BOMB':
            events.append(TRIED_BOMB_BUT_NOT_POSSIBLE)
            self.logger.debug(f'Successfully appended event {TRIED_BOMB_BUT_NOT_POSSIBLE}')
        
    #action_list[0] = self_action


    #calc new_betas (except for first round)

        self.model[:,index] = self.model[:,index] + alpha * state_to_features(old_game_state) * (reward_from_events(self, events) + gamma * (np.matmul(state_to_features(old_game_state),self.model)).argmax(axis=0) +  np.matmul(state_to_features(new_game_state), self.model[:,index]) )
        
        #betas normieren
        self.model = self.model * (1/np.sum(self.model, axis=1))[:, np.newaxis]
        
        self.logger.debug(f'Weights successfully updated in step {new_game_state["step"]}')



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    index = action_to_index(last_action)

    self.transitions.pop()
    old_game_state = self.transitions.pop().new_s

    
    #self.model[:,index] = self.model[:,index] + alpha * state_to_features(old_game_state) * (reward_from_events(self, events) + gamma * (np.matmul(state_to_features(old_game_state),self.model)).argmax(axis=0) +  np.matmul(state_to_features(last_game_state), self.model[:,index]) )
    
    #self.model[:,index] = self.model[:,index] + alpha * state_to_features(old_game_state) * (reward_from_events(self, events) + gamma * (np.matmul(state_to_features(old_game_state),self.model)).argmax(axis=0) +  np.matmul(state_to_features(last_game_state), self.model[:,index]) )
        
    self.logger.debug(f'Weights successfully updated in final step')

    #betas normieren
    self.model = self.model * (1/np.sum(self.model, axis=1))[:, np.newaxis]

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
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: 0,
        e.BOMB_DROPPED: -9,
        e.WAITED: -9,

        #reward for valid movement
        e.MOVED_UP: 3,
        e.MOVED_RIGHT: 3,
        e.MOVED_DOWN: 3,
        e.MOVED_LEFT: 3,
        
        # e.COIN_COLLECTED: 1,
        # e.KILLED_OPPONENT: 5,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
        MOVED_OUT_OF_BOMB_RANGE: 10,
        MOVED_INTO_BOMB_RANGE: -10,
        GOES_AWAY_FROM_BOMB: self.num,

        MOVED_AGAINST_WALL: -1,

        TRIED_BOMB_BUT_NOT_POSSIBLE: -10,

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



#my definition

def action_to_index(self_action) -> int:

    action_index = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

    return action_index[self_action]


def get_bombs_in_range(self_coord,bombs_coord):
    return bombs_coord[( (bombs_coord[:,0] == self_coord[0]) & ((bombs_coord[:,1] >= self_coord[1] - 3) | (bombs_coord[:,1] <= self_coord[1] + 3))  ) | ( (bombs_coord[:,1] == self_coord[1]) & ((bombs_coord[:,0] >= self_coord[0] - 3) | (bombs_coord[:,0] <= self_coord[0] + 3)) )]


def goes_away_from_bomb(self_coord, bombs_in_range, self_coord_new):
    
    if bombs_in_range.size == 0:
        return False, 0

    else:
        #coord that changed
        a = np.nonzero(self_coord_new - self_coord)[0][0]
        #coord that didnt change
        b = np.where((self_coord_new - self_coord) == 0)[0][0]


        #weighted score of number of moved away bombs
        num = 0


        #get points for moving out of range (moved orthogonally to bomb)
        num += np.count_nonzero(bombs_in_range[:,b] != self_coord[a])

        #moved parallel to bomb

        #remaining bombs
        remaining_bombs = bombs_in_range[bombs_in_range[:,b] == self_coord[b]]

        #give reward for moving away from bombs
        num += 1 * ((abs(remaining_bombs[:,a] - self_coord_new[a]) - abs(remaining_bombs[:,a] - self_coord[a])) == 1).sum()

        #give penalty for moving towards ... number of bombs
        num -= 5 * ((abs(remaining_bombs[:,a] - self_coord_new[a]) - abs(remaining_bombs[:,a] - self_coord[a])) == -1).sum()

        return True, num

    



import torch
import numpy as np


class ReplayBuffer:
    
    def __init__(self, size = 1000):
        """
        the replay buffer contains data which is described by the (state, action, reward, state') 4-tupel in theory.
        """
        self._replay_list_state1  = []
        self._replay_list_actions = []
        self._replay_ten_rewards = torch.zeros((size, 1), dtype=torch.float32)
        self._replay_list_state2  = []
        self._size = 0
        self._next_index = 0 # this is the index where to input the next element if add_transition() is called
        self.max_size = size


    def add_transition(self, state1, actions, reward, state2):
        if self._size < self.max_size:
            self._size += 1
        else:
            self._replay_list_actions.pop(0)
            self._replay_list_state1.pop(0)
            self._replay_list_state2.pop(0)
        self._replay_list_state1.append(state1)
        self._replay_list_actions.append(actions)
        self._replay_ten_rewards[self._next_index] = reward
        self._replay_list_state2.append(state2)
        self._next_index = self._next_index + 1 % self.max_size


    def sample_minibatch(self, batch_size):
        indexes = np.random.randint(0, self._size, batch_size)
        output_list_st1  = []
        output_list_acti = []
        output_list_st2  = []
        for i in indexes:
            output_list_st1.append( self._replay_list_state1[i] )
            output_list_acti.append(self._replay_list_actions[i])
            output_list_st2.append( self._replay_list_state2[i] )
        return output_list_st1, output_list_acti, self._replay_ten_rewards[indexes, :], output_list_st2


    def get_buffer_size(self):
        return self._size


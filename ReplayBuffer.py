
import torch
import numpy as np


class ReplayBuffer:
    
    def __init__(self, size = 1000, number_agents = 1):
        """
        the replay buffer contains data which is described by the (state, action, reward, state') 4-tupel in theory.
        """
        self._replay_list_state1_agent_inp  = [ [] for _ in range(number_agents) ]
        self._replay_list_state1_critic_inp = [ [] for _ in range(number_agents) ]
        self._replay_list_state2_agent_inp  = [ [] for _ in range(number_agents)  ]
        self._replay_list_state2_critic_inp = [ [] for _ in range(number_agents) ]
        self._replay_list_actions_agent_outp= [ [] for _ in range(number_agents)  ]
        self._replay_list_actions_critic_merged_inp = [ [] for _ in range(number_agents) ]
        self._replay_ten_rewards = torch.zeros((size, 1), dtype=torch.float32)
        self._size = 0
        self._next_index = 0 # this is the index where to input the next element if add_transition() is called
        self.max_size = size
        self._number_agents  = number_agents


    def add_transition(self, state1_agents_inp, state1_critics_inp, 
                       actions_agents_outp, actions_critics_merged_inp,
                       reward,
                       state2_agents_inp, state2_critics_inp):
        if self._size < self.max_size:
            self._size += 1
        else:
            for lst in [self._replay_list_state1_agent_inp,
                        self._replay_list_state1_critic_inp,
                        self._replay_list_state2_agent_inp,
                        self._replay_list_state2_critic_inp,
                        self._replay_list_actions_agent_outp,
                        self._replay_list_actions_critic_merged_inp]:
                for agent_or_critic in lst:
                    agent_or_critic.pop(0)

        for target, source in zip(self._replay_list_state1_agent_inp,
                                  state1_agents_inp):
            target.append(source)
        for target, source in zip(self._replay_list_state1_critic_inp,
                                  state1_critics_inp):
            target.append(source)
        for target, source in zip(self._replay_list_actions_agent_outp,
                                  actions_agents_outp):
            target.append(source)
        for target, source in zip(self._replay_list_actions_critic_merged_inp,
                                  actions_critics_merged_inp):
            target.append(source)
        for target, source in zip(self._replay_list_state2_agent_inp,
                                  state2_agents_inp):
            target.append(source)
        for target, source in zip(self._replay_list_state2_critic_inp,
                                  state2_critics_inp):
            target.append(source)

        self._replay_ten_rewards[self._next_index] = reward
        self._next_index = self._next_index + 1 % self.max_size


    def sample_minibatch(self, batch_size):
        indexes = np.random.randint(0, self._size, batch_size)
        olst_ag_inp1  = [ [] for _ in range(number_agents) ]
        olst_cr_inp1  = [ [] for _ in range(number_agents) ]
        olst_ag_out   = [ [] for _ in range(number_agents) ]
        olst_cr_ac_in = [ [] for _ in range(number_agents) ]
        olst_ag_inp2  = [ [] for _ in range(number_agents) ]
        olst_cr_inp2  = [ [] for _ in range(number_agents) ]
        for i in indexes:
            olst_ag_inp1.append(  self._replay_list_state1_agent_inp[i] )
            olst_cr_inp1.append(  self._replay_list_state1_critic_inp[i] )
            olst_ag_out.append(   self._replay_list_actions_agent_outp[i] )
            olst_cr_ac_in.append( self._replay_list_actions_critic_merged_inp[i] )
            olst_ag_inp2.append(  self._replay_list_state2_agent_inp[i] )
            olst_cr_inp2.append(  self._replay_list_state2_critic_inp[i] )
        return (olst_ag_inp1, olst_cr_inp1),
               (olst_ag_out, olst_cr_ac_in),
               self._replay_ten_rewards[indexes, :],
               (olst_ag_inp2, olst_cr_inp2)


    def get_buffer_size(self):
        return self._size


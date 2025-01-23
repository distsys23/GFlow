"""
Function for building buffer, where some trained data would be saved.

Using:
numpy: 1.22.4
"""
import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, actor_dims, critic_dims,
                 n_agents, n_actions, batch_size):
        """
        :param max_size: number for max size for storing transition
        :param critic_dims: number of dimensions for the critic
        :param actor_dims: number of dimensions for the actor
        :param n_actions: number of actions
        :param n_agents: number of agents
        :param batch_size: number of batch size
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.task_state_memory = [None] * self.mem_size
        self.VM_state_memory = [None] * self.mem_size
        self.edge_index_memory = [None] * self.mem_size
        self.task_state_next_memory = [None] * self.mem_size
        self.VM_state_next_memory = [None] * self.mem_size
        self.edge_index_next_memory = [None] * self.mem_size
        self.actions_memory = np.zeros((self.mem_size, n_agents))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

    def store_transition(self, task_state, VM_state, edge_index, actions, reward, task_state_next, VM_state_next, edge_index_next, done): 
        """
        :param raw_obs: state raw observations
        :param state:
        :param action:
        :param reward:
        :param raw_obs_: new state raw observations
        :param state_: new states
        :param done: terminal flags
        """
        index = self.mem_cntr % self.mem_size

        self.task_state_memory[index] = task_state
        self.VM_state_memory[index] = VM_state
        self.edge_index_memory[index] = edge_index
        self.actions_memory[index] = actions
        self.reward_memory[index] = reward
        self.task_state_next_memory[index] = task_state_next
        self.VM_state_next_memory[index] = VM_state_next
        self.edge_index_next_memory[index] = edge_index_next
        self.terminal_memory[index] = done
        self.mem_cntr += 1


    def sample_buffer(self):
        """
        :return:  appropriate memories
                  actor_states: individual arrays of states
                  states: flattened combination of state arrays
                  actions: flattened combination of action arrays
                  rewards: individual arrays of rewards
                  actor_new_states: flattened combination of new action arrays
                  states_: individual arrays of new states
                  terminal: individual arrays of terminal flags
        """
        max_mem = min(self.mem_cntr, self.mem_size)  # current memory size
        # memories could not be selected multiple times (replace=False)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        task_state = [self.task_state_memory[i] for i in batch]
        vm_state = [self.VM_state_memory[i] for i in batch]
        edge_index = [self.edge_index_memory[i] for i in batch]
        actions = self.actions_memory[batch]
        reward = self.reward_memory[batch]
        task_state_next = [self.task_state_next_memory[i] for i in batch]
        vm_state_next = [self.VM_state_next_memory[i] for i in batch]
        edge_index_next = [self.edge_index_next_memory[i] for i in batch]
        terminal = self.terminal_memory[batch]
        
        return task_state, vm_state, edge_index, actions, reward, task_state_next, vm_state_next, edge_index_next, terminal

    def ready(self):
        """
        check memory state
        :return: memory state
                 Ture:  fill up the batch size
        """
        if self.mem_cntr >= self.batch_size:
            return True

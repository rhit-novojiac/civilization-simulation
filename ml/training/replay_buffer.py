import random
import torch

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # We assume states and next_states are already tensors. If not, they must be converted before returning.
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

import random
import torch
import torch.nn as nn
import torch.optim as optim
from ml.training.replay_buffer import ReplayBuffer

class DQNTrainer:
    def __init__(self, policy_net, target_net, learning_rate=1e-4, capacity=50000, batch_size=128, gamma=0.99, dual_headed=False, sync_every=1000):
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.dual_headed = dual_headed
        self.loss_fn = nn.MSELoss()
        
        self.steps_done = 0
        self.sync_every = sync_every

    def push_transition(self, state, action, reward, next_state, done):
        """Saves a transition to memory."""
        self.memory.push(state, action, reward, next_state, done)

    @staticmethod
    def save_models(brains, filepath="data/models/"):
        """Saves Macro and Micro policy networks for all species to disk."""
        import os
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
        for species_id, brain_dict in brains.items():
            if 'macro' in brain_dict:
                macro_path = os.path.join(filepath, f"species_{species_id}_macro.pt")
                torch.save(brain_dict['macro'].state_dict(), macro_path)
            if 'micro' in brain_dict:
                micro_path = os.path.join(filepath, f"species_{species_id}_micro.pt")
                torch.save(brain_dict['micro'].state_dict(), micro_path)

    def optimize_step(self):
        """
        Performs one step of Double-DQN optimization.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch of transitions
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert lists of tensors to single stacked tensors
        states_batch = torch.stack(states)
        rewards_batch = torch.tensor(rewards, dtype=torch.float32)
        next_states_batch = torch.stack(next_states)
        dones_batch = torch.tensor(dones, dtype=torch.float32)
        
        if self.dual_headed:
            move_actions = [a[0] for a in actions]
            stance_actions = [a[1] for a in actions]
            move_actions_batch = torch.tensor(move_actions, dtype=torch.long)
            stance_actions_batch = torch.tensor(stance_actions, dtype=torch.long)
            
            # 1. Compute Q(s, a) using Policy Net
            move_q, stance_q = self.policy_net(states_batch)
            move_val = move_q.gather(1, move_actions_batch.unsqueeze(1)).squeeze(1)
            stance_val = stance_q.gather(1, stance_actions_batch.unsqueeze(1)).squeeze(1)

            # 2. Double-DQN target evaluation:
            with torch.no_grad():
                next_move_q, next_stance_q = self.policy_net(next_states_batch)
                best_next_move = next_move_q.argmax(dim=1, keepdim=True)
                best_next_stance = next_stance_q.argmax(dim=1, keepdim=True)
                
                t_move_q, t_stance_q = self.target_net(next_states_batch)
                next_move_val = t_move_q.gather(1, best_next_move).squeeze(1)
                next_stance_val = t_stance_q.gather(1, best_next_stance).squeeze(1)
                
                # Independent Bellman Targets
                expected_move_val = rewards_batch + (self.gamma * next_move_val * (1.0 - dones_batch))
                expected_stance_val = rewards_batch + (self.gamma * next_stance_val * (1.0 - dones_batch))

            # 3. Compute loss and optimize
            loss_move = self.loss_fn(move_val, expected_move_val)
            loss_stance = self.loss_fn(stance_val, expected_stance_val)
            total_loss = loss_move + loss_stance
        else:
            actions_batch = torch.tensor(actions, dtype=torch.long)

            # 1. Compute Q(s, a) using Policy Net
            state_action_values = self.policy_net(states_batch).gather(1, actions_batch.unsqueeze(1)).squeeze(1)

            # 2. Double-DQN target evaluation:
            with torch.no_grad():
                best_next_actions = self.policy_net(next_states_batch).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states_batch).gather(1, best_next_actions).squeeze(1)
                expected_state_action_values = rewards_batch + (self.gamma * next_q_values * (1.0 - dones_batch))

            # 3. Compute loss and optimize
            total_loss = self.loss_fn(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clamp gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Soft update target network
        self.steps_done += 1
        if self.steps_done % self.sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return total_loss.item()

def get_decayed_epsilon(tick, start_epsilon=1.0, min_epsilon=0.05, decay_ticks=100000):
    """
    Computes linearly decayed epsilon.
    """
    if tick >= decay_ticks:
        return min_epsilon
    return start_epsilon - (start_epsilon - min_epsilon) * (float(tick) / float(decay_ticks))

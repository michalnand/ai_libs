import torch



class DiscreteReplayBuffer:
    def __init__(self, buffer_size, state_shape, num_actions, dtype=torch.float32):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.dtype = dtype
        self.clear()

    def clear(self):
        self.states  = torch.zeros((self.buffer_size,) + self.state_shape, dtype=self.dtype)
        self.actions = torch.zeros((self.buffer_size,), dtype=torch.int32)
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.dones   = torch.zeros((self.buffer_size,), dtype=torch.bool)
        self.ptr     = 0
        self.size    = 0

    def add(self, state, reward, action, done):
        idx = self.ptr % self.buffer_size
        
        # Ensure inputs are tensors
        self.states[idx]  = torch.as_tensor(state, dtype=self.dtype)
        self.rewards[idx] = torch.as_tensor(reward, dtype=torch.float32)
        self.actions[idx] = torch.as_tensor(action, dtype=torch.int32)
        self.dones[idx]   = torch.as_tensor(done, dtype=torch.bool)

        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)

    def sample_sequence(self, batch_size, horizon, device):
        # We need sequences of length (horizon + 1) to get next_states
        max_valid_idx = self.size - horizon - 1
        if max_valid_idx <= 0:
            raise ValueError("Not enough data in buffer to sample sequences.")
            
        # Sample random starting indices
        start_idxs = torch.randint(0, max_valid_idx, (batch_size,))
        
        # Create a grid of indices: Shape [batch_size, horizon + 1]
        seq_idxs = start_idxs.unsqueeze(1) + torch.arange(horizon + 1)
        
        # Gather sequences and send to device
        s_seq = self.states[seq_idxs].to(device)   # [B, H+1, dim_s]
        a_seq = self.actions[seq_idxs].to(device)  # [B, H+1, dim_a]
        r_seq = self.rewards[seq_idxs].to(device)  # [B, H+1]
        d_seq = self.dones[seq_idxs].to(device)    # [B, H+1]
        
        return s_seq, a_seq, r_seq, d_seq

    

class ContinuousReplayBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, dtype=torch.float32):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dtype = dtype
        self.clear()

    def clear(self):
        self.states  = torch.zeros((self.buffer_size,) + self.state_shape, dtype=self.dtype)
        self.actions = torch.zeros((self.buffer_size,) + self.action_shape, dtype=torch.float32)
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32)
        self.dones   = torch.zeros((self.buffer_size,), dtype=torch.bool)
        self.ptr     = 0
        self.size    = 0

    def add(self, state, reward, action, done):
        idx = self.ptr % self.buffer_size
        
        # Ensure inputs are tensors
        self.states[idx]  = torch.as_tensor(state, dtype=self.dtype)
        self.rewards[idx] = torch.as_tensor(reward, dtype=torch.float32)
        self.actions[idx] = torch.as_tensor(action, dtype=torch.float32)
        self.dones[idx]   = torch.as_tensor(done, dtype=torch.bool)

        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)

    def sample_sequence(self, batch_size, horizon, device):
        # We need sequences of length (horizon + 1) to get next_states
        max_valid_idx = self.size - horizon - 1
        if max_valid_idx <= 0:
            raise ValueError("Not enough data in buffer to sample sequences.")
            
        # Sample random starting indices
        start_idxs = torch.randint(0, max_valid_idx, (batch_size,))
        
        # Create a grid of indices: Shape [batch_size, horizon + 1]
        seq_idxs = start_idxs.unsqueeze(1) + torch.arange(horizon + 1)
        
        # Gather sequences and send to device
        s_seq = self.states[seq_idxs].to(device)   # [B, H+1, dim_s]
        a_seq = self.actions[seq_idxs].to(device)  # [B, H+1, dim_a]
        r_seq = self.rewards[seq_idxs].to(device)  # [B, H+1]
        d_seq = self.dones[seq_idxs].to(device)    # [B, H+1]
        
        return s_seq, a_seq, r_seq, d_seq
import torch


class DiscretePPOBuffer:

    def __init__(self, buffer_size, num_envs, state_shape):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.state_shape = (state_shape if isinstance(state_shape, tuple) else (state_shape,))

        # Stored on CPU RAM to avoid GPU memory overhead
        self.states = torch.zeros(
            (buffer_size, num_envs) + self.state_shape, dtype=torch.float32
        )
        self.actions = torch.zeros(
            (buffer_size, num_envs), dtype=torch.long
        )  # Categorical actions are integer indices
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.log_probs = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32
        )
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32)

        self.advantages = torch.zeros(
            (buffer_size, num_envs), dtype=torch.float32
        )
        self.returns = torch.zeros((buffer_size, num_envs), dtype=torch.float32)

        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, state, action, reward, done, log_prob, value):
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.long)  # Must be int64/long for PyTorch Categorical
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32)
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, last_done, gamma=0.99, gae_lambda=0.95):
        """Calculates GAE backwards across the rollout window."""
        last_gae_lam = 0.0

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - torch.as_tensor(
                    last_done, dtype=torch.float32
                )
                next_values = torch.as_tensor(last_value, dtype=torch.float32)
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    def get_batch(self, indices, device="cuda"):
        """Slices the CPU buffer by flat indices and transfers the mini-batch to target device."""
        b_states = self.states.reshape((-1,) + self.state_shape)
        b_actions = self.actions.reshape(-1)  # 1D array of action integers
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return (
            b_states[indices].to(device, non_blocking=True),
            b_actions[indices].to(device, non_blocking=True),
            b_log_probs[indices].to(device, non_blocking=True),
            b_advantages[indices].to(device, non_blocking=True),
            b_returns[indices].to(device, non_blocking=True),
            b_values[indices].to(device, non_blocking=True),
        )

    def get_generator(self, mini_batch_size, device="cuda"):
        total_samples = self.buffer_size * self.num_envs
        indices = torch.randperm(total_samples)  # CPU shuffle

        for start_idx in range(0, total_samples, mini_batch_size):
            mb_indices = indices[start_idx : start_idx + mini_batch_size]
            yield self.get_batch(mb_indices, device=device)



class ContinuousPPOBuffer:

    def __init__(self, buffer_size, num_envs, state_shape, num_actions):
        self.buffer_size    = buffer_size
        self.num_envs       = num_envs

        self.state_shape    = (state_shape if isinstance(state_shape, tuple) else (state_shape,))
        self.num_actions    = num_actions

        # Always store on CPU RAM to conserve GPU VRAM
        self.states = torch.zeros((buffer_size, num_envs) + self.state_shape, dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, num_envs, num_actions), dtype=torch.float32)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32)

        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size, num_envs), dtype=torch.float32)

        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, state, action, reward, done, log_prob, value):
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32)
        self.ptr += 1

    def compute_returns_and_advantages(
        self, last_value, last_done, gamma=0.99, gae_lambda=0.95
    ):
        last_gae_lam = 0.0

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - torch.as_tensor(
                    last_done, dtype=torch.float32
                )
                next_values = torch.as_tensor(last_value, dtype=torch.float32)
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    # -------------------------------------------------------------------------
    # Option A: Direct Index Sampler (No Generator)
    # -------------------------------------------------------------------------
    def get_batch(self, indices, device="cuda"):
        """Fetches specific indices from flattened buffer and moves only that mini-batch to GPU."""
        # Flatten time and env dimensions on CPU
        b_states = self.states.reshape((-1,) + self.state_shape)
        b_actions = self.actions.reshape((-1, self.num_actions))
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Slice on CPU, then transfer to GPU
        return (
            b_states[indices].to(device, non_blocking=True),
            b_actions[indices].to(device, non_blocking=True),
            b_log_probs[indices].to(device, non_blocking=True),
            b_advantages[indices].to(device, non_blocking=True),
            b_returns[indices].to(device, non_blocking=True),
            b_values[indices].to(device, non_blocking=True),
        )

    # -------------------------------------------------------------------------
    # Option B: Generator (Transfers per yield to GPU)
    # -------------------------------------------------------------------------
    def get_generator(self, mini_batch_size, device="cuda"):
        total_samples = self.buffer_size * self.num_envs
        indices = torch.randperm(total_samples)  # CPU shuffle

        for start_idx in range(0, total_samples, mini_batch_size):
            mb_indices = indices[start_idx : start_idx + mini_batch_size]
            yield self.get_batch(mb_indices, device=device)
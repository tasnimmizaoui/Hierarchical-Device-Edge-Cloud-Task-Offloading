"""Independent DQN (replay, epsilon-greedy, target network, checkpoints)."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int = 2, seed: int = 42,
                 gamma: float = .94, learning_rate: float = 7e-4,
                 batch_size: int = 64, memory_size: int = 20_000,
                 target_interval: int = 100):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.online = QNetwork(obs_dim, n_actions)
        self.target = QNetwork(obs_dim, n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.gamma, self.batch_size = gamma, batch_size
        self.target_interval = target_interval
        self.learn_steps = 0
        self.decision_steps = 0
        self._recent_losses: list[float] = []

    def epsilon(self) -> float:
        return .05 + (.90 - .05) * np.exp(-self.decision_steps / 1100.0)

    def act(self, observation: np.ndarray, training: bool = False) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.obs_dim,) or not np.isfinite(observation).all():
            raise ValueError("Invalid observation for DQN agent")
        if training:
            self.decision_steps += 1
            if self.rng.random() < self.epsilon():
                return self.rng.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
            return int(self.online(x).argmax(dim=1).item())

    def observe(self, transition: Transition) -> float | None:
        state = np.asarray(transition.state, dtype=np.float32)
        nxt = np.asarray(transition.next_state, dtype=np.float32)
        if state.shape != (self.obs_dim,) or nxt.shape != (self.obs_dim,):
            raise ValueError("Transition observation shape differs from agent schema")
        if not np.isfinite(state).all() or not np.isfinite(nxt).all():
            raise ValueError("Transition observations must be finite")
        if not (0 <= transition.action < self.n_actions):
            raise ValueError("Transition action is out of range")
        if not np.isfinite(transition.reward):
            raise ValueError("Transition reward must be finite")
        self.memory.append(Transition(state.copy(), transition.action,
                                      float(transition.reward), nxt.copy(),
                                      bool(transition.done)))
        loss = self.learn()
        if loss is not None:
            self._recent_losses.append(loss)
        return loss

    def pop_training_losses(self) -> list[float]:
        """Return and clear optimization losses for logging at episode boundaries."""
        losses, self._recent_losses = self._recent_losses, []
        return losses

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = self.rng.sample(self.memory, self.batch_size)
        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32)
        done = torch.tensor([t.done for t in batch], dtype=torch.float32)
        predicted = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Double DQN: online action selection, target value evaluation.
            best = self.online(next_states).argmax(dim=1, keepdim=True)
            target = rewards + self.gamma * (1.0 - done) * (
                self.target(next_states).gather(1, best).squeeze(1)
            )
        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': self.online.state_dict(), 'obs_dim': self.obs_dim,
                    'n_actions': self.n_actions, 'decision_steps': self.decision_steps,
                    'learn_steps': self.learn_steps, 'algorithm': 'Double-DQN',
                    'gamma': self.gamma}, p)

    @classmethod
    def load(cls, path: str | Path) -> 'DQNAgent':
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(int(checkpoint['obs_dim']), int(checkpoint['n_actions']))
        model.online.load_state_dict(checkpoint['state_dict'])
        model.target.load_state_dict(checkpoint['state_dict'])
        model.decision_steps = int(checkpoint.get('decision_steps', 0))
        model.learn_steps = int(checkpoint.get('learn_steps', 0))
        model.online.eval()
        return model


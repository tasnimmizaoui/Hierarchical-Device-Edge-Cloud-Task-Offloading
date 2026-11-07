"""Independent DQN (replay, epsilon-greedy, target network, checkpoints)."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
                 gamma: float = .94, learning_rate: float = 3e-4,
                 batch_size: int = 64, memory_size: int = 8_000,
                 target_interval: int = 500, epsilon_decay_steps: int = 15_000,
                 epsilon_start: float = .90, epsilon_end: float = .05,
                 algorithm: str = "double-dqn"):
        if epsilon_decay_steps <= 0 or not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Invalid epsilon schedule")
        if algorithm not in {"dqn", "double-dqn"}:
            raise ValueError("algorithm must be 'dqn' or 'double-dqn'")
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
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.algorithm = algorithm
        self.learn_steps = 0
        self.decision_steps = 0
        self.episode_epsilon: float | None = None
        self._recent_losses: list[float] = []

    def epsilon(self) -> float:
        if self.episode_epsilon is not None:
            return self.episode_epsilon
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.decision_steps / self.epsilon_decay_steps
        )

    def act(self, observation: np.ndarray, training: bool = False) -> int:
        action, _, _ = self.act_with_diagnostics(observation, training=training)
        return action

    def act_with_diagnostics(
        self, observation: np.ndarray, training: bool = False
    ) -> tuple[int, np.ndarray, float]:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.obs_dim,) or not np.isfinite(observation).all():
            raise ValueError("Invalid observation for DQN agent")
        epsilon = self.epsilon()
        if training:
            self.decision_steps += 1
            explore = self.rng.random() < epsilon
        else:
            explore = False
        with torch.no_grad():
            x = torch.tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
            q_values = self.online(x).squeeze(0).cpu().numpy().copy()
        greedy_action = int(np.argmax(q_values))
        action = self.rng.randrange(self.n_actions) if explore else greedy_action
        return action, q_values, epsilon

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
            if self.algorithm == "double-dqn":
                best = self.online(next_states).argmax(dim=1, keepdim=True)
                next_values = self.target(next_states).gather(1, best).squeeze(1)
            else:
                next_values = self.target(next_states).max(dim=1).values
            target = rewards + self.gamma * (1.0 - done) * (
                next_values
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
                    'learn_steps': self.learn_steps, 'algorithm': self.algorithm,
                    'gamma': self.gamma, 'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size, 'memory_size': self.memory_size,
                    'target_interval': self.target_interval,
                    'epsilon_decay_steps': self.epsilon_decay_steps,
                    'epsilon_start': self.epsilon_start,
                    'epsilon_end': self.epsilon_end}, p)

    @classmethod
    def load(cls, path: str | Path) -> DQNAgent:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(
            int(checkpoint['obs_dim']),
            int(checkpoint['n_actions']),
            gamma=float(checkpoint.get('gamma', .94)),
            learning_rate=float(checkpoint.get('learning_rate', 3e-4)),
            batch_size=int(checkpoint.get('batch_size', 64)),
            memory_size=int(checkpoint.get('memory_size', 8_000)),
            target_interval=int(checkpoint.get('target_interval', 500)),
            epsilon_decay_steps=int(checkpoint.get('epsilon_decay_steps', 15_000)),
            epsilon_start=float(checkpoint.get('epsilon_start', .90)),
            epsilon_end=float(checkpoint.get('epsilon_end', .05)),
            algorithm=str(checkpoint.get('algorithm', 'double-dqn')).lower(),
        )
        model.online.load_state_dict(checkpoint['state_dict'])
        model.target.load_state_dict(checkpoint['state_dict'])
        model.decision_steps = int(checkpoint.get('decision_steps', 0))
        model.learn_steps = int(checkpoint.get('learn_steps', 0))
        model.online.eval()
        return model

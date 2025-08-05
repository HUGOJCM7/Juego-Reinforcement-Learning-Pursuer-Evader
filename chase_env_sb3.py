import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm

class Agent:
    """Representa a un agente en el entorno, con física básica de movimiento."""

    def __init__(self, mass: float = 1.0, max_force: float = 0.05, radius: float = 0.05, color: str = "blue"):
        self.mass = mass
        self.max_force = max_force
        self.radius = radius
        self.color = color
        self.pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.vel: np.ndarray = np.zeros(2, dtype=np.float32)
        self.force: float = 0.0
        self.angle: float = 0.0
        self.reset()

    def reset(self) -> None:
        """Reinicia la posición, velocidad y fuerza del agente a valores iniciales."""
        self.pos = np.random.uniform(-0.8, 0.8, size=2).astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.force = 0.0
        self.angle = 0.0

    def apply_force(self, magnitude: float, angle: float) -> None:
        """Aplica una fuerza recortada al agente en la dirección dada por el ángulo."""
        self.force = np.clip(magnitude, 0.0, self.max_force)
        self.angle = angle
        fx = self.force * np.cos(self.angle)
        fy = self.force * np.sin(self.angle)
        acc = np.array([fx, fy], dtype=np.float32) / self.mass
        self.vel += acc

    def update(self, dt: float, bounds: Tuple[float, float]) -> None:
        """Actualiza la posición del agente con colisiones simples contra los límites del entorno."""
        self.pos += self.vel * dt
        for i in range(2):
            lo, hi = bounds
            if self.pos[i] - self.radius < lo:
                self.pos[i] = lo + self.radius
                self.vel[i] = 0.0
            elif self.pos[i] + self.radius > hi:
                self.pos[i] = hi - self.radius
                self.vel[i] = 0.0

class ChaseEnv(gym.Env):
    """
    Entorno de persecución 2D continuo para el aprendizaje por refuerzo.
    Un agente (RL) interactúa con un bot que sigue una estrategia simple o un modelo entrenado.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        role: str = "evader",
        bot_model: Optional[BaseAlgorithm] = None,
        render_mode: Optional[str] = None,
        dt: float = 0.1,
        max_steps: int = 200
    ):
        super().__init__()
        assert role in ("evader", "pursuer"), "El rol debe ser 'evader' o 'pursuer'"

        self.role = role
        self.bot_model = bot_model
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.bounds = (-1.0, 1.0)

        # Definición de agentes y roles
        if self.role == "evader":
            self.agent_rl = Agent(radius=0.05, color="green")
            self.bot = Agent(radius=0.07, color="red")
        else: # pursuer
            self.agent_rl = Agent(radius=0.07, color="red")
            self.bot = Agent(radius=0.05, color="green")

        # Definición de espacios de observación y acción
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -np.pi], dtype=np.float32),
            high=np.array([self.agent_rl.max_force, np.pi], dtype=np.float32),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.agent_rl.reset()
        self.bot.reset()
        self.step_count = 0
        
        while np.linalg.norm(self.agent_rl.pos - self.bot.pos) < self.agent_rl.radius + self.bot.radius:
            self.agent_rl.reset()
            self.bot.reset()
            
        initial_info = {
            "initial_agent_pos": self.agent_rl.pos,
            "initial_bot_pos": self.bot.pos
        }
        return self._get_obs(), initial_info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.agent_rl.pos, self.bot.pos]).astype(np.float32)

    def _get_bot_action(self) -> Tuple[float, float]:
        """Obtiene la acción del bot usando un modelo o una estrategia heurística."""
        if self.bot_model:
            obs_bot = np.concatenate([self.bot.pos, self.agent_rl.pos]).astype(np.float32)
            bot_action, _ = self.bot_model.predict(obs_bot, deterministic=True)
            force, angle = np.clip(bot_action, self.action_space.low, self.action_space.high)
        else:
            direction = self.agent_rl.pos - self.bot.pos
            angle = np.arctan2(direction[1], direction[0])
            if self.role == "evader":
                angle += np.pi
            force = self.bot.max_force
            
        return force, angle

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        
        force, angle = np.clip(action, self.action_space.low, self.action_space.high)
        self.agent_rl.apply_force(force, angle)

        bot_force, bot_angle = self._get_bot_action()
        self.bot.apply_force(bot_force, bot_angle)

        self.agent_rl.update(self.dt, self.bounds)
        self.bot.update(self.dt, self.bounds)

        dist = np.linalg.norm(self.agent_rl.pos - self.bot.pos)
        caught = dist < (self.agent_rl.radius + self.bot.radius)

        if self.role == "evader":
            reward = (self.dt) - 1000.0 * float(caught)
        else:
            reward = 1000.0 * float(caught) - (self.dt)
        reward -= self.dt * self.agent_rl.force

        terminated = caught or (self.step_count >= self.max_steps)
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

class EvadeEnv(ChaseEnv):
    def __init__(self, bot_model: Optional[BaseAlgorithm] = None, render_mode: Optional[str] = None, **kwargs):
        super().__init__(role="evader", bot_model=bot_model, render_mode=render_mode, **kwargs)

class PursueEnv(ChaseEnv):
    def __init__(self, bot_model: Optional[BaseAlgorithm] = None, render_mode: Optional[str] = None, **kwargs):
        super().__init__(role="pursuer", bot_model=bot_model, render_mode=render_mode, **kwargs)

__all__ = ["EvadeEnv", "PursueEnv", "ChaseEnv"]
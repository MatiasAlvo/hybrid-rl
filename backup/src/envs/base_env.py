from abc import ABC, abstractmethod
import gymnasium as gym

class BaseEnvironment(gym.Env, ABC):
    """Base class for all environments."""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def step(self, action):
        """Execute one time step within the environment."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the environment to an initial state."""
        pass

    @abstractmethod
    def render(self):
        """Render the environment."""
        pass

from src.envs.base_env import BaseEnvironment
from src.envs.inventory.simulator import Simulator
from src import gym, np

class InventoryEnv(BaseEnvironment):
    """Inventory environment implementing the BaseEnvironment interface."""
    
    def __init__(self):
        super().__init__()
        self.simulator = Simulator()
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.simulator.state_dim,),
                dtype=np.float32
            )
        })
        
    def step(self, action):
        """Execute one time step within the environment."""
        next_state, reward, done, info = self.simulator.step(action)
        return next_state, reward, done, False, info  # False is for truncated in gymnasium
        
    def reset(self, seed=None):
        """Reset the environment to an initial state."""
        if seed is not None:
            super().reset(seed=seed)
        initial_state = self.simulator.reset()
        return initial_state, {}  # Empty dict is for info in gymnasium
        
    def render(self):
        """Render the environment."""
        pass

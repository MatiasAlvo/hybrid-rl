from torchrl.envs import EnvBase
from tensordict import TensorDict
from src.envs.inventory.simulator import Simulator


class InventoryEnv(EnvBase):
    def __init__(self, args):
        super().__init__()
        self.simulator = Simulator(args.device)
        self.device = args.device
        self.observation_params = args.observation_params
        self.problem_params = args.problem_params
        
    def _step(self, tensordict):
        action = tensordict["action"]
        
        # Run simulator step and get detailed info
        next_state, reward, done, info = self.simulator.step(
            action, 
            self.observation_params,
            self.problem_params
        )
        
        # Store everything needed for both HDPO and PPO
        return tensordict.update({
            "next_observation": next_state,
            "reward": reward,
            "done": done,
            "total_cost": info["total_cost"],
            "state": info["state"],
            "action_log_prob": info.get("action_log_prob", None)
        })
        
    def _reset(self, tensordict):
        state = self.simulator.reset()
        return tensordict.update({"observation": state})

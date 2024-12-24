from torchrl.collectors import SyncDataCollector
from tensordict.tensordict import TensorDict

class InventoryCollector:
    def __init__(self, env, policy, args):
        self.collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=args.batch_size,
            reset_at_each_iter=True,
            storing_keys=[
                "observation",
                "action",
                "reward",
                "next",
                "done"
            ]
        )
        self.args = args

    def collect_batch(self):
        return next(self.collector)

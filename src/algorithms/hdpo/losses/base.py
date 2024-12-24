from torchrl.objectives import LossModule

class InventoryLoss(LossModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, tensordict):
        raise NotImplementedError 
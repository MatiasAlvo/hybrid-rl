from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Tuple

class BaseAlgorithm(ABC):
    """Base interface for RL algorithms."""
    
    @abstractmethod
    def collect_experience(self, env, policy) -> Dict[str, Any]:
        """Collect training data in algorithm-specific way.
        
        Args:
            env: Environment to collect experience from
            policy: Policy to use for collection
            
        Returns:
            Dictionary containing collected experience
        """
        pass
    
    @abstractmethod
    def compute_loss(self, experience: Dict[str, Any]) -> torch.Tensor:
        """Compute loss based on collected experience.
        
        Args:
            experience: Dictionary containing collected experience
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update policy using algorithm-specific approach.
        
        Args:
            experience: Dictionary containing collected experience
            
        Returns:
            Dictionary containing update statistics
        """
        pass

    def save(self, path: str) -> None:
        """Save algorithm state."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load algorithm state."""
        self.load_state_dict(torch.load(path))

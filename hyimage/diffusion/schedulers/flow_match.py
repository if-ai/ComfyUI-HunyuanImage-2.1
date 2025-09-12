"""
Flow Matching Scheduler for HunyuanImage
Implements the flow matching algorithm for denoising
"""

import torch
import numpy as np
from typing import Optional, Union, List


class FlowMatchScheduler:
    """
    Flow Matching scheduler for HunyuanImage denoising
    """
    
    def __init__(
        self,
        solver: str = "dpm_solver",
        shift: int = 4,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        guidance_rescale: float = 0.0
    ):
        """
        Initialize the Flow Match Scheduler
        
        Args:
            solver: Solver type ('dpm_solver' or 'euler')
            shift: Time shift parameter
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            guidance_rescale: Guidance rescaling factor
        """
        self.solver = solver
        self.shift = shift
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        
        # Generate timesteps
        self.timesteps = self._generate_timesteps()
        self.current_step = 0
        
    def _generate_timesteps(self) -> torch.Tensor:
        """Generate timesteps for the denoising process"""
        # Linear spacing from 1 to 0
        timesteps = torch.linspace(1.0, 0.0, self.num_inference_steps + 1)
        
        # Apply shift transformation
        if self.shift > 0:
            # Sigmoid-based time shifting for better sampling
            shift_factor = self.shift / 10.0
            timesteps = torch.sigmoid((timesteps - 0.5) * shift_factor * 2) 
            timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min())
        
        return timesteps[:-1]  # Remove the last timestep (0)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        Perform one step of the denoising process
        
        Args:
            model_output: Output from the diffusion model
            timestep: Current timestep
            sample: Current sample/latent
            generator: Random generator for stochastic sampling
            return_dict: Whether to return a dictionary
            
        Returns:
            Updated sample after denoising step
        """
        # Get current and next timestep
        idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            # If exact match not found, find closest
            idx = torch.argmin(torch.abs(self.timesteps - timestep))
        else:
            idx = idx[0]
            
        t_curr = self.timesteps[idx]
        t_next = self.timesteps[idx + 1] if idx + 1 < len(self.timesteps) else torch.tensor(0.0)
        
        # Calculate step size
        dt = t_next - t_curr
        
        if self.solver == "euler":
            # Euler solver: x_{t+dt} = x_t + dt * v_t
            sample = sample + dt * model_output
            
        elif self.solver == "dpm_solver":
            # DPM-Solver for better quality
            # This is a simplified version - full DPM-Solver would cache previous predictions
            if hasattr(self, '_prev_output') and self._prev_output is not None:
                # Second-order update when we have previous output
                sample = sample + dt * (1.5 * model_output - 0.5 * self._prev_output)
            else:
                # First-order update for first step
                sample = sample + dt * model_output
            
            # Cache current output for next step
            self._prev_output = model_output.clone()
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        if return_dict:
            return {"prev_sample": sample}
        return sample
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples for a given timestep (used for img2img)
        
        Args:
            original_samples: Original clean samples
            noise: Random noise
            timesteps: Timestep values
            
        Returns:
            Noisy samples
        """
        # Flow matching: interpolate between noise and data
        # x_t = t * noise + (1 - t) * data
        timesteps = timesteps.view(-1, 1, 1, 1)
        noisy_samples = timesteps * noise + (1 - timesteps) * original_samples
        return noisy_samples
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu"):
        """
        Set the number of inference steps and regenerate timesteps
        
        Args:
            num_inference_steps: Number of denoising steps
            device: Device to place timesteps on
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = self._generate_timesteps().to(device)
        self.current_step = 0
        self._prev_output = None  # Reset cached output
    
    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        """
        Scale model input if needed (for compatibility)
        Flow matching doesn't require scaling, so we return as-is
        """
        return sample
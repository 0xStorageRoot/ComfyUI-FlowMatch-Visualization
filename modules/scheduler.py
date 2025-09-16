import numpy as np
from typing import Optional, Tuple, Dict, Union, List

class FlowUniPCMultistepScheduler:
    final_sigmas_type = "zero"
    num_train_timesteps = 1000
    use_dynamic_shifting = False
    shift = 5     

    # Modified from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.set_timesteps
    # just for visualization
    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas

        if not self.use_dynamic_shifting:
            sigmas = shift * sigmas / (1 +(shift - 1) * sigmas) 

        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]  # pyright: ignore

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        timesteps = sigmas * self.num_train_timesteps
        return timesteps


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma

class FlowDPMSolverMultistepScheduler:
    num_train_timesteps = 1000
    use_dynamic_shifting = False
    shift = 5
    final_sigmas_type = "zero"

    # Modified from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.set_timesteps
    # just for visualization
    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        shift = self.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas) 

        timesteps = sigmas * self.num_train_timesteps
        self.timesteps = timesteps

        return timesteps
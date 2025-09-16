#  Package Modules
import os
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time

#  ComfyUI Modules
import folder_paths
from comfy.utils import ProgressBar

#  Your Modules
from .modules.calculator import CalculatorSteps


#  Basic practice to get paths from ComfyUI
custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "my-custom-nodes")
custom_nodes_output_dir = os.path.join(folder_paths.get_output_directory(), "my-custom-nodes")

class FlowMatchSteps:
    #  Define the input parameters of the node here.
    @classmethod
    def INPUT_TYPES(s):
        return {
            #  If the key is "required", the value must be filled.
            "required": {
                "scheduler_style": (
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {"default": "Flow"},
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "shift": (
                    "INT", {"default": 5, "min": 1, "max": 100, "step": 1}
                ),
                "boundary": (
                    "FLOAT", {"default": 0.900, "min": 0.00, "max": 1.00, "step": 0.001}
                ),
                "skip": (
                    "INT", {"default": 0, "min": 0, "max": 100, "step": 1}
                ),
                "cfg_skip": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}
                ),
            },
            #  If the key is "optional", the value is optional.
            "optional": {
            }
        }

    #  Define these constants inside the node.
    #  `RETURN_TYPES` is important, as it limits the parameter types that can be passed to the next node, in `INPUT_TYPES()` above.
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    #  `FUNCTION` is the function name that will be called in the node.
    FUNCTION = "calculate_steps"
    #  `CATEGORY` is the category name that will be used when user searches the node.
    CATEGORY = "FlowMatchSteps"

    #  In the function, use same parameter names as you specified in `INPUT_TYPES()`
    def calculate_steps(self,
                        scheduler_style: str,
                        steps: int,
                        shift: int,
                        boundary: float,
                        skip: int,
                        cfg_skip: float,
                        ) -> Tuple[int,int,int]:
        result = CalculatorSteps.calculate_steps(
            scheduler_style=scheduler_style,
            steps=steps,
            shift=shift,
            boundary=boundary,
            skip=skip,
            cfg_skip=cfg_skip,
        )

        image = CalculatorSteps.plot_step_info(
            result, 
            "Step Info Visualization Test",
        )

        return image

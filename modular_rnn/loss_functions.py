from typing import Union

import numpy as np
import torch.nn as nn
import torch

from .models import ModelOutput


mse = nn.MSELoss()

class MSEOnlyLoss(nn.Module):
    def forward(self,
                model_outputs: dict[str, Union[np.ndarray, torch.Tensor]],
                target_outputs: dict[str, np.ndarray],
                masks: dict[str, Union[np.ndarray, torch.Tensor]],
                excluded_outputs: list[str]):

        error = 0.
        for output_name in target_outputs.keys():
            if output_name in excluded_outputs:
                continue

            mask = masks.get(output_name,
                             np.ones_like(model_outputs[output_name]))
            #mask = masks[output_name] if output_name in masks else np.ones_like(model_outputs[output_name])

            error += mse(model_outputs[output_name] * mask, target_outputs[output_name] * mask)

        return error
        
        
class TolerantLoss(nn.Module):
    def __init__(self,
                 tolerance_in_deg: float,
                 direction_output_name: str):
        super().__init__()
        self.tolerance_in_deg = tolerance_in_deg
        self.tolerance_in_rad = np.deg2rad(self.tolerance_in_deg)
        self.direction_output_name = direction_output_name
        

    def forward(self,
                model_outputs: dict[str, ModelOutput],
                target_outputs: dict[str, Union[np.ndarray, torch.Tensor]],
                masks: dict[str, Union[np.ndarray, torch.Tensor]]):

        model_dir_output = model_outputs[self.direction_output_name].as_tensor()
        target_dir_output = target_outputs[self.direction_output_name]
        dir_mask = masks[self.direction_output_name]

        model_angle  = torch.atan2(model_dir_output[:, :, 1], model_dir_output[:, :, 0])
        target_angle = torch.atan2(target_dir_output[:, :, 1], target_dir_output[:, :, 0])

        angle_incorrect = torch.abs(model_angle - target_angle) > self.tolerance_in_rad

        error = mse(angle_incorrect.T * dir_mask.T * target_dir_output.T,
                    angle_incorrect.T * dir_mask.T * model_dir_output.T)
        
        return error

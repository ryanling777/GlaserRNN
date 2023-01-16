from typing import Union

import numpy as np
import torch.nn as nn
import torch

from modules import ModelOutput


mse = nn.MSELoss()

class MSEOnlyLoss(nn.Module):
    def __init__(self,
                 output_names: list[str]):
        super().__init__()
        self.output_names = output_names
        
    def forward(self,
                model_outputs: dict[str, Union[np.ndarray, torch.Tensor]],
                target_outputs: dict[str, np.ndarray],
                masks: dict[str, Union[np.ndarray, torch.Tensor]]):
        
        error = 0.
        for output_name in self.output_names:
            
            mask = masks.get(output_name,
                             torch.ones(model_outputs[output_name].shape))
            #print(mask)
            #mask = masks[output_name] if output_name in masks else np.ones_like(model_outputs[output_name])

            error += mse(model_outputs[output_name].as_tensor() * mask, target_outputs[output_name] * mask)
        #print( "error is ")
        #print(error)
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

        error = mse(angle_incorrect * torch.permute(dir_mask, (2, 0, 1)) * torch.permute(target_dir_output, (2, 0, 1)),
                    angle_incorrect * torch.permute(dir_mask, (2, 0, 1)) * torch.permute(model_dir_output, (2, 0, 1)))
        
        return error

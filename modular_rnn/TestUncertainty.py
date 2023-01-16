"This file is to test out whether the Center Out task is doing what it should be doing"

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F  

import matplotlib.pyplot as plt
import seaborn as sns  

torch.set_num_threads(1)

from connections import ConnectionConfig
from tasks import CossinUncertaintyTaskWithReachProfiles
from training import train

# Set parameters
tau = 100

# Timestep of the simulation
dt = 5

alpha = dt/tau

tolerance = 5

noise = 0.05

# activation function of the neurons
nonlin_fn = F.relu

# length of each trial
L = 1200

#number of trials in a batch
batch_size = 1

# Instantiate the task
task = CossinUncertaintyTaskWithReachProfiles(dt, tau, L, batch_size)

# Generate a trial
batch_inputs, batch_outputs, batch_masks, trial_params = task.get_trial_batch()


#print(trial_params)
print(batch_outputs)
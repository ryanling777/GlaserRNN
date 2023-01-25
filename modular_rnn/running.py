import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F 
from PyalData.pyaldata.interval import restrict_to_interval 

import matplotlib.pyplot as plt
  

torch.set_num_threads(1)

# Create task and RNN

from connections import ConnectionConfig
from models import MultiRegionRNN
from loss_functions import TolerantLoss, MSEOnlyLoss
from tasks import CossinUncertaintyTaskWithReachProfiles
from tasks import CenterOut
from training import train

# For Testing

from testing import run_test_batches

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
batch_size = 64
# batch_size = 64

# special loss for the uncertainty task
#loss_fn = TolerantLoss(tolerance, 'hand')
loss_fn = MSEOnlyLoss(['hand'])

# Create Task
task = CenterOut(dt, tau, L, batch_size)
#task = CossinUncertaintyTaskWithReachProfiles(dt, tau, L, batch_size)

# dictionary defining the modules in the RNN
# single region called motor_cortex

regions_config_Dict = {
    'motor_cortex' : {
        'n_neurons' : 50,
        'alpha' : alpha,
        'p_rec' : 1., 
        'dynamics_noise' : noise,
    
    }
}

# name and dimensionality of the outputs we want the RNN to 
# hand will have an x and y coordinate, with time as the 3rd axis.
output_dims = {'hand': 2}

rnn = MultiRegionRNN(task.N_in, 
    output_dims,
    alpha,
    nonlin_fn,
    regions_config_Dict,
    connection_configs = [],
    input_configs = [
        ConnectionConfig('inputs','motor_cortex')
    ],
    output_configs = [
        ConnectionConfig('motor_cortex', 'hand')
        ]
)

# Training the model

losses = train(rnn, task, 500, loss_fn) #500
#print(losses)
plt.plot(losses[10:])
plt.title('Loss vs Trials')
plt.xlabel('Trials')
plt.ylabel('MSE')
plt.savefig('Loss.png')


# Testing the model 
# function in pyal called restrict_to_interval to cut out from trial start to end
# every row is a trial
test_df = run_test_batches(10, rnn, task)
test_df.to_pickle('/home/rnl18/modular_rnn/modular_rnn/SpeedAnalysis/Test_OutputUnrestricted.pkl')
test_df = restrict_to_interval(test_df,"idx_trial_start","idx_trial_end",0,0,None,None,None,False, True, 'hand_model_output')
#ax = sns.scatterplot(x = 'target_dir', y = 'endpoint_location', data = test_df, palette = 'tab10')
#ax.set_aspect('equal')
#plt.show()
#
#print(test_df.head(5))
test_df.to_pickle('/home/rnl18/modular_rnn/modular_rnn/SpeedAnalysis/Test_Output.pkl')
plt.savefig('Accuracy.png')


fig, ax = plt.subplots()

# arr is matrix for each iteration
for arr in test_df.hand_model_output.values[:100]:
    ax.scatter(*arr.T, alpha = 0.1, color = 'tab:blue')
    print(*arr.T)
    
ax.set_title('model output')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.savefig('Model Output.png')
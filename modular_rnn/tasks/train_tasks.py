from .base_task import Task

import numpy as np
import scipy.stats

from .reach_profile import extent_curve, speed_curve

def calc_cue_spread(arr: np.ndarray) -> float:
    return np.max(np.diff(np.sort(arr)))

class CenterOut(Task):
    def __init__(self, dt, tau, T, N_batch, N_targets=8, stim_noise=0.05):
        
        # define output_dims - the hand will hold cosx and siny, max euclidean distance is 1.
        output_dims = {'hand' : 2}

        # 3 columns for input - cosx, siny and cue.
        super().__init__(3, output_dims, dt, tau, T, N_batch)
        self.N_targets = N_targets
        self.stim_noise = stim_noise 
        


    def generate_trial_params(self, batch, trial):
        tid = np.random.randint(0, self.N_targets)
        target_dir = (2*np.pi/self.N_targets) * tid

        params = dict()
        params['target_id'] = tid
        params['target_dir'] = target_dir
        params['target_cos'] = np.cos(target_dir)
        params['target_sin'] = np.sin(target_dir)
        params['target_cossin'] = np.array([params['target_cos'], params['target_sin']])

        # to be randomised later
        params['idx_trial_start'] = 50
        params['idx_target_on'] = params['idx_trial_start'] + np.random.randint(50, 200)
        #params['idx_go_cue'] = 500
        #params['idx_trial_end'] = 900
        params['idx_go_cue']      = params['idx_target_on'] + np.random.randint(200, 500)
        params['idx_trial_end']   = params['idx_go_cue'] + 450




        params['stim_noise'] = self.stim_noise * np.random.randn(self.T, self.N_in)

        return params

    def trial_function(self, time, params):
        target_cossin = params['target_cossin']

        #start with just noise
        #input_signal = self.stim_noise * np.random.randn(self.N_in)
        input_signal = params['stim_noise'][time, :]

        # add 0 to third column representing go cue
        if time >= params['idx_target_on']:
            input_signal = np.append(target_cossin, np.zeros(1))

         # go signal should be on after the go cue, last signal is non-zero for go cue
        if time >= params['idx_go_cue']:
            input_signal += np.append(np.zeros(2), 1)

        output_signal = {}

        
        if time < params['idx_go_cue']:
            output_signal['hand'] = np.zeros(self.output_dims['hand'])
        else:
            shifted_time = time - params['idx_go_cue']

            # position is the extent projected to the x and y axes
            extent_at_t = extent_curve(shifted_time)
            # stores speed at the shifted time.
            speed_at_t = speed_curve(shifted_time)
            
            output_signal['hand'] = target_cossin * extent_at_t
        
        
        #if time < params['idx_go_cue']:
        #   output_signal = np.zeros(self.N_out)
        #else:
        #   output_signal = target_cossin

        masks_t = {}
        if params['idx_trial_start'] < time < params['idx_trial_end']:
            masks_t['hand'] = np.ones(self.output_dims['hand'])
        else:
            masks_t['hand'] = np.zeros(self.output_dims['hand'])

        return input_signal, output_signal, masks_t





class CossinUncertaintyTaskWithReachProfiles(Task):
    def __init__(self, dt, tau, T, N_batch, target_kappa=25, stim_noise=0, cue_kappas=(5, 50)):

        output_dims = {'hand' : 2,
                       'uncertainty' : 1}

        # 11 inputs because you would have 5 initial which each have a cos and sin
        super().__init__(11,
                         output_dims,
                         dt,
                         tau,
                         T,
                         N_batch)
        self.stim_noise = stim_noise
        self.target_kappa = target_kappa
        self.cue_kappas = cue_kappas
        
    def generate_trial_params(self, batch, trial):
        params = dict()
        
        target_dir = (np.pi/2) + np.random.vonmises(mu = 0, kappa = self.target_kappa)
        params['target_dir'] = target_dir
        params['target_cos'] = np.cos(target_dir)
        params['target_sin'] = np.sin(target_dir)
        params['target_cossin'] = np.array([params['target_cos'], params['target_sin']])
        
        cue_kappa = np.random.choice(self.cue_kappas)
        params['cue_kappa'] = cue_kappa
        params['cue_slice_locations'] = np.sort(np.random.vonmises(mu = target_dir, kappa = cue_kappa, size = self.N_in // 2))
        params['cue_var'] = scipy.stats.circvar(params['cue_slice_locations'])
        params['cue_var_log'] = np.log(params['cue_var'])
        params['cue_spread'] = calc_cue_spread(params['cue_slice_locations'])
        
        params['idx_trial_start'] = 50
        params['idx_target_on']   = params['idx_trial_start'] + np.random.randint(50, 200)
        params['idx_go_cue']      = params['idx_target_on'] + np.random.randint(200, 500)
        params['idx_trial_end']   = params['idx_go_cue'] + 450

        params['stim_noise'] = self.stim_noise * np.random.randn(self.T, self.N_in)
        params['cue_input'] = np.array([np.cos(ang) for ang in params['cue_slice_locations']] + [np.sin(ang) for ang in params['cue_slice_locations']] + [0.])
        
        return params
    
    def trial_function(self, time, params):
        target_cossin = params['target_cossin']
        
        # start with just noise
        input_signal = params['stim_noise'][time, :]
        
        # add the input after the target onset
        #if params['idx_target_on'] <= time < params['idx_target_on']+20:
        if params['idx_target_on'] <= time:
            input_signal += params['cue_input']

        # go signal should be on after the go cue, last signal is non-zero for go cue
        if time >= params['idx_go_cue']:
            input_signal += np.append(np.zeros(10), 1)
            
        # in the beginning the output is nothing, then it's the mean position or velocity profile
        outputs_t = {}
        if time < params['idx_go_cue']:
            outputs_t['hand'] = np.zeros(self.output_dims['hand'])
        else:
            shifted_time = time - params['idx_go_cue']

            # position is the extent projected to the x and y axes
            extent_at_t = extent_curve(shifted_time)
            outputs_t['hand'] = target_cossin * extent_at_t
            
        # we always care about correct position
        masks_t = {}
        if time > params['idx_trial_start']:
            masks_t['hand'] = np.ones(self.output_dims['hand'])
        else:
            masks_t['hand'] = np.zeros(self.output_dims['hand'])
            
        outputs_t['uncertainty'] = params['cue_spread']
        masks_t['uncertainty'] = 1.

        return input_signal, outputs_t, masks_t

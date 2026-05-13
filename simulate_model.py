import torch
from torch import nn
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
import os 
import argparse 
import pandas as pd

from pykalman import KalmanFilter
from scipy.signal import savgol_filter

from PIL import Image 
import imageio

from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
import sys
from model import UDiffPlan, loss_func_UDiffPlan
from plot_func import getTrack
import shutil

# from config_pred import Config_pred

from config_nn import Config_NN

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from PIL import Image
import os
import imageio 
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from sklearn import svm
from scipy.interpolate import interp1d

import math
from train_func import data_prepare, CustomDataset, \
    create_data_dict, split_data_dict, apply_preprocessing, apply_normalization, dict_to_tensor, \
    create_metrics, update_metrics, print_metrics, process_traj_for_plot, preprocess_batch

np.random.seed(555)
torch.manual_seed(555)

## features
## x, y, psi, vx, vy, omega, delta, D, theta, ec
## we should use delta, D of EV but not that of TV. (control inputs)


class Load_Model():
    def __init__(self, args):
        super(Load_Model, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = Config_NN()
        self.config.is_train = False

        self.features_EV = self.config.features_EV
        self.features_TV = self.config.features_TV
        self.x_fidx = self.config.x_fidx 
        self.y_fidx = self.config.y_fidx 

            
        # FEATURES = 7
        self.OUTPUT_DIM = self.config.output_dim ### x, y
        self.SEQUENCE_LENGTH = self.config.sequence_length
        self.LABELS = self.config.labels

        self.save_dir = self.config.save_dir 
        self.args = args
        self.times = []

        self.unet_params = self.config.unet_params
        self.beta_schedule_params = self.config.beta_schedule_params

        #! DATA PREPROCESSING (DATA_PREPROCESSING_MULTIMODAL)
        self.DIFF = UDiffPlan(self.config, self.device, self.unet_params, self.beta_schedule_params, 'sr3', self.args).to(self.device)
        self.DIFF.set_new_noise_schedule(device=self.device, phase='train')
        loss_fn = torch.nn.MSELoss()
        self.DIFF.set_loss(loss_fn)
        self.DIFF_state_dict = torch.load(self.save_dir+'/UDiffPlan_state_dict.pth')['model_state_dict']
        self.DIFF.load_state_dict(self.DIFF_state_dict)
        self.DIFF.args = self.args

        self.DIFF.eval()

        for param in self.DIFF.parameters():
            param.requires_grad = False 
            
    def load_model(self, state_EV, EV_kappa, decision_EV, mode_EV, state_TV, TV_kappa, decision_TV, mode_TV, leader, who, step):

        state_EV = np.array(state_EV)
        decision_EV = np.array(decision_EV)
        state_TV = np.array(state_TV)
        decision_TV = np.array(decision_TV)
        mode_EV = np.array(mode_EV)
        mode_TV = np.array(mode_TV)

        leader = np.array(leader)
        
        # TV_data_res = resampling(state_TV, 0)
        # EV_data_res = resampling(state_EV, 0)
        #! NO RESAMPLING IS NEEDED, SINCE THE MODEL IS TRAINED WITH TIME-BASED DATA
        #! (60, 10)
        TV_data_res = state_TV
        EV_data_res = state_EV

        Y = np.array([decision_TV, 1, decision_EV, 1], dtype=int) #! style == 1 (4,)
        
        label = np.concatenate([mode_TV[np.newaxis], mode_EV[np.newaxis]], axis=0) #(2, 4)

        total_TV_past = np.expand_dims(TV_data_res, axis=0)
        total_TV_past = np.delete(total_TV_past, [6, 7], axis=2)
        total_EV_past = np.expand_dims(EV_data_res, axis=0)
        
        total_TV_kappa = np.expand_dims(TV_kappa, axis=0)
        total_EV_kappa = np.expand_dims(EV_kappa, axis=0)
        
        total_Y = np.expand_dims(Y, axis=0) #(batch, 2)
        total_Y_prob = np.expand_dims(label, axis=0).astype(float) #(batch, 2, 4)

        total_leader = np.expand_dims(leader, axis=0)

        total_EV_past_org =  total_EV_past.copy()
        total_TV_past_org =  total_TV_past.copy()
        # total_TV_future_org =  total_TV_future.copy() 

        total_EV_future = None
        total_TV_future = None

        total_data = create_data_dict(
            total_EV_past, total_EV_future, total_EV_kappa,
            total_TV_past, total_TV_future, total_TV_kappa,
            total_Y, total_Y_prob, total_leader
        )

        total_data = apply_preprocessing(total_data, self.config)

        total_data, norm_value = apply_normalization(total_data, self.config, mode='test')

        total_data = dict_to_tensor(total_data, self.device)


        real_EV_past = total_data['EV']['past'].permute(0, 2, 1) 
        real_EV_kappa = total_data['EV']['kappa']
        real_TV_past = total_data['TV']['past'].permute(0, 2, 1) 
        real_TV_kappa = total_data['TV']['kappa'] 

        real_Y = total_data['Y']
        real_Y_prob = total_data['Y_prob']
        real_leader = total_data['leader']

        init_future_org_EV_past = total_data['EV']['past_org'].permute(0, 2, 1)
        init_future_org_TV_past = total_data['TV']['past_org'].permute(0, 2, 1)
        TV_past_init = total_data['TV']['past_init'].permute(0, 2, 1)

        real_EV_past = torch.nan_to_num(real_EV_past)
        real_TV_past = torch.nan_to_num(real_TV_past)
        
        condition = real_Y[:, 1].unsqueeze(1).repeat(1, self.SEQUENCE_LENGTH).unsqueeze(1)
        norm_value = norm_value['TV']

        #! calculate inference time
        with torch.no_grad():
            if who == 'EV':
                loss, pred_traj_samples, pred_Y, pred_traj_mean, pred_traj_log_var, _, _ = self.DIFF(real_EV_past, None, \
                                real_EV_kappa, real_TV_past, None, real_TV_kappa, condition,\
                                init_future_org_EV_past, init_future_org_TV_past, norm_value,\
                                mode='generate', diff_mode='test')



            data_idx = 0
            traj = process_traj_for_plot(
                    real_EV_past, None, real_TV_past, None, pred_traj_mean, pred_traj_samples,
                    data_idx, norm_value,
                    TV_past_init, real_Y, pred_Y, real_leader, self.config
            )


            pred_TV_future_x, pred_TV_future_y = traj['pred']  #(60, 2)
            pred_TV_future_sample_x, pred_TV_future_sample_y = traj['pred_samples']

            pred_TV_future = np.stack([pred_TV_future_x, pred_TV_future_y], axis=1)  #(60, 2)
            pred_TV_future_sample = np.stack([pred_TV_future_sample_x, pred_TV_future_sample_y], axis=2)

            #!pred_TV_future should be (2, 60, 2) (trajectory number, sequence, states)
            #!pred_label should be (2)
            pred_Y = pred_Y.squeeze(0)

            #! when using stable, meann, variance from diffusion model represents latent vectors' mean and variance
            sampled_traj_mean = pred_TV_future_sample.mean(axis=0)
            sampled_traj_log_var = pred_TV_future_sample.std(axis=0, ddof=0)

            return pred_TV_future, pred_Y.cpu().detach().numpy(), pred_TV_future_sample, sampled_traj_mean, sampled_traj_log_var ##(2, N_samples, 60, 2)


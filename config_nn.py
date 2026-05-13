###config
class Config_NN:
    def __init__(self):
        
        #### PATH
        self.data_dir = 'data'
        self.data_processed_dir = 'data_processed'
        self.map_file = 'LMS_Track.txt'
        self.save_dir = 'train'
        self.save_test_dir = 'test'
        self.pretrained_dir = 'train'

        #### DATA CONFIGURATION
        self.x_fidx = 0
        self.y_fidx = 1
        self.vx_fidx = 3
        self.vy_fidx = 4
        self.features_EV = ['x', 'y', 'psi', 'vx', 'vy', 'omega', 'delta', 'D', 'theta', 'ec']
        self.features_TV = ['x', 'y', 'psi', 'vx', 'vy', 'omega', 'theta', 'ec']
        self.kappa_len = 51 # kappa length

        self.sequence_length = 60
        self.interval = 40
        self.n_classes = 4

        self.max_samples_phase1 = 5000
        self.max_samples_phase2 = 5000

        self.onehot = 1

        self.track_width = 0.25
        self.dt = 0.05
        # Driving mode labels: 0=following, 1=overtaking, 2=driving, 3=blocking
        self.label_name = ['following', 'overtaking', 'driving', 'blocking']

        #### MODEL TRAINING CONFIGURATION
        self.output_dim = 2 # x, y positions
        self.num_epochs = 50
        self.test_size = 0.1
        self.batch_size = 4
        self.optim_lr = 0.0001

        #### MODEL INPUT
        self.input_dim = len(self.features_EV) + len(self.features_TV)+2+1+2
        self.input_concat = 2

        #### DIFFUSION MODEL PARAMETERS
        self.unet_params = {
            "in_channel": 25, 
            "out_channel": 2,
            "inner_channel": 64,
            "channel_mults": [
                1,
                2,
            ],
            "attn_res": [
                15,
            ],
            "res_blocks": 1,
            "dropout": 0.1,
            "norm_groups": 4,
            "image_size": 60,
        }

        self.beta_schedule_params = {
            "train": {
                "schedule": "cosine",
                "n_timestep": 10,
                "linear_start": 1e-7,
                "linear_end": 0.01
            },
            "test": {
                "schedule": 'cosine',
                "n_timestep": 10, 
                "linear_start": 1e-7,
                "linear_end": 0.01
            }
        }
        self.N_samples = 16
        
        #### AUXILIARY ENCODER PARAMETERS
        self.labels = 2
        self.chns= [32, 16, 8]
        self.k_E = [3, 3, 3, 3] # Encoder kernel
        self.s_E = [2, 2, 2, 2] # Encoder stride
        self.p_E = [1, 1, 1, 1] # Encoder padding
        self.latent_dim = 128 # bottleneck latent dimension
        self.dropout_E = 0.3 # Encoder dropout

        ### Savitzky-Golay Filter
        self.filter_window = 21
        self.filter_order = 3

        ### Evaluation Horizons (in timesteps, at 20Hz: 20=1s, 40=2s, 60=3s)
        self.eval_horizons = [20, 40, 60]

        ### To Divide training / simulation
        self.is_train = True # Default, train for model train/test

        
        
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

from tqdm import tqdm

from model import UDiffPlan, loss_func_UDiffPlan
from plot_func import plot_visualization
from train_func import CustomDataset, evaluation, \
    create_data_dict, split_data_dict, apply_preprocessing, apply_normalization, dict_to_tensor, \
    create_metrics, update_metrics, print_metrics, process_traj_for_plot, preprocess_batch, run_batch
from config_nn import Config_NN


np.random.seed(555)
torch.manual_seed(555)
torch.cuda.manual_seed_all(555)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', '-s', default=1, type=int)
    parser.add_argument('--gpu', '-g', default=0, type=int)
    parser.add_argument('--r', '-r', default='n', type=str, help='resume training?')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = Config_NN()

    EPOCH = config.num_epochs
    TEST_SIZE = config.test_size
    SEQUENCE_LENGTH = config.sequence_length
    BATCH_SIZE = config.batch_size

    DATA_DIR = config.data_processed_dir
    SAVE_DIR = config.save_dir
    SAVE_TEST_DIR = config.save_test_dir

    UNET_PARAMS = config.unet_params
    BETA_SCHEDULE_PARAMS = config.beta_schedule_params
    OPTIM_LR = config.optim_lr

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(SAVE_TEST_DIR):
        os.makedirs(SAVE_TEST_DIR)


    total_EV_past = np.load(DATA_DIR+'/'+'EV_past.npy', allow_pickle=True)
    total_EV_future = np.load(DATA_DIR+'/'+'EV_future.npy', allow_pickle=True)
    total_EV_kappa = np.load(DATA_DIR+'/'+'EV_kappa.npy', allow_pickle=True)

    total_TV_past = np.load(DATA_DIR+'/'+'TV_past.npy', allow_pickle=True)
    total_TV_future = np.load(DATA_DIR+'/'+'TV_future.npy', allow_pickle=True)
    total_TV_kappa = np.load(DATA_DIR+'/'+'TV_kappa.npy', allow_pickle=True)

    total_leader = np.load(DATA_DIR+'/'+'leader.npy', allow_pickle=True)
    total_Y = np.load(DATA_DIR+'/'+'Y.npy', allow_pickle=True)
    total_Y_prob = np.load(DATA_DIR+'/'+'Y_prob.npy', allow_pickle=True)

    total_road = np.load(DATA_DIR+'/'+'road.npy', allow_pickle=True)

    total_TV_past = np.delete(total_TV_past, [6, 7], axis=2) # delete TV's control input data
    total_TV_future = np.delete(total_TV_future, [6, 7], axis=2) # delete TV's control input data

    # =============================================================================
    # Data Pipeline
    # =============================================================================
    total_data = create_data_dict(
        total_EV_past, total_EV_future, total_EV_kappa,
        total_TV_past, total_TV_future, total_TV_kappa,
        total_Y, total_Y_prob, total_leader, total_road
    )

    total_data, test_data = split_data_dict(total_data, TEST_SIZE)

    total_data = apply_preprocessing(total_data, config)
    test_data = apply_preprocessing(test_data, config)

    total_data, normalizing_values_total = apply_normalization(total_data, config, mode='train')
    test_data, normalizing_values_test = apply_normalization(test_data, config, mode='test')

    train_data, val_data = split_data_dict(total_data, TEST_SIZE)

    train_data = dict_to_tensor(train_data, device)
    val_data = dict_to_tensor(val_data, device)
    test_data = dict_to_tensor(test_data, device)

    # =============================================================================
    # Create Datasets and DataLoaders
    # =============================================================================
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=True)

    loss_fn = torch.nn.MSELoss()
    DIFF = UDiffPlan(config, device, UNET_PARAMS, BETA_SCHEDULE_PARAMS, 'sr3', args).to(device)
    DIFF.set_new_noise_schedule(device=device, phase='train')
    DIFF.set_loss(loss_fn)
    loss_func = loss_func_UDiffPlan

    if args.step == 1:
        dataloader = train_dataloader 
        batch_size = BATCH_SIZE
        norm_value = normalizing_values_total['TV']
        dataloader_len = len(dataloader)
        optimizer = torch.optim.Adam(DIFF.parameters(), lr=OPTIM_LR)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, \
            lr_lambda=lambda epoch:0.95**epoch, last_epoch=-1, verbose=False)
        
    elif args.step == 2:
        dataloader = test_dataloader
        batch_size = len(test_dataset)
        dataloader_len = len(dataloader)
        norm_value = normalizing_values_test['TV']
        EPOCH = 1
        for param in DIFF.parameters():
            param.requires_grad = False

    # For testing or resuming training
    if args.step == 2 or args.r == 'y':
        DIFF_state_dict = torch.load(SAVE_DIR+'/UDiffPlan_state_dict.pth')['model_state_dict']
        DIFF.load_state_dict(DIFF_state_dict)
        DIFF.args = args  # Update to current args instead of loaded model's args

    total_loss = []
    total_loss_traj = []
    total_loss_label = []

    total_loss_val = []
    total_loss_traj_val = []
    total_loss_label_val = []

    BEST_LOSS = 1000

    # Training loop
    for epoch in range(EPOCH):
        print('epoch: ', epoch)
        batch_loss = 0
        batch_loss_traj = 0
        batch_loss_label = 0

        metrics_train = create_metrics()
        metrics_val = create_metrics()

        diff_mode = 'pretrain' if args.step == 1 else 'test'

        with tqdm(dataloader, unit='batch') as tepoch:
            for idx, batch_data in enumerate(tepoch):
                result = run_batch(DIFF, batch_data, norm_value, loss_func, config, diff_mode, seq_length=SEQUENCE_LENGTH)
                b = result['batch']
                loss, loss_traj, loss_label = result['loss'], result['loss_traj'], result['loss_label']

                if args.step == 1:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    tepoch.set_postfix(loss=loss.item())

                if args.step == 2:
                    for data_idx in range(BATCH_SIZE):
                        traj = process_traj_for_plot(
                            b['EV_past'], b['EV_future'], b['TV_past'], b['TV_future'],
                            result['pred_traj_mean'], result['pred_traj_samples'],
                            data_idx, norm_value, b['TV_past_init'], b['Y'], result['pred_Y'], b['leader'], config
                        )
                        plot_visualization(
                            traj['EV_past'][0], traj['EV_past'][1],
                            traj['TV_past'][0], traj['TV_past'][1],
                            traj['EV_future'][0], traj['EV_future'][1],
                            traj['TV_future'][0], traj['TV_future'][1],
                            traj['pred'][0], traj['pred'][1],
                            traj['real_Y'], traj['pred_Y'], config
                        )
                        if data_idx % 2 == 0:
                            TV_str = str(b['Y'][data_idx, 0].cpu().detach().numpy())
                            EV_str = str(b['Y'][data_idx, 1].cpu().detach().numpy())
                            plt.savefig(SAVE_TEST_DIR + f'/test_{data_idx}_{TV_str}_{EV_str}.png')
                            plt.close()

                metrics_batch = evaluation(b['TV_future'], result['pred_traj_mean'], norm_value, b['TV_past_init'], result['batch_size'], dataloader_len, config)
                update_metrics(metrics_train, metrics_batch)

                batch_loss += loss / result['batch_size']
                batch_loss_traj += loss_traj / result['batch_size']
                batch_loss_label += loss_label / result['batch_size']
        
        curr_loss = batch_loss/dataloader_len
        total_loss.append((curr_loss).item())

        curr_loss_traj = batch_loss_traj / dataloader_len
        curr_loss_label = batch_loss_label / dataloader_len
        total_loss_traj.append((curr_loss_traj).item())
        total_loss_label.append((curr_loss_label).item())

        if args.step == 1:
            scheduler.step()

        # Set model to evaluation mode
        DIFF.eval()

        with torch.no_grad():
            batch_loss_val = 0
            batch_loss_traj_val = 0
            batch_loss_label_val = 0

            for val_idx, batch_data_val in enumerate(val_dataloader):
                result_val = run_batch(DIFF, batch_data_val, norm_value, loss_func, config, diff_mode)
                b_val = result_val['batch']

                batch_loss_val += result_val['loss']
                batch_loss_traj_val += result_val['loss_traj'] / result_val['batch_size']
                batch_loss_label_val += result_val['loss_label'] / result_val['batch_size']

                metrics_batch_val = evaluation(
                    b_val['TV_future'], result_val['pred_traj_mean'], norm_value,
                    b_val['TV_past_init'], result_val['batch_size'], len(val_dataloader), config
                )
                update_metrics(metrics_val, metrics_batch_val)

            print_metrics(metrics_val, prefix='_val')

            curr_loss_val = batch_loss_val / len(val_dataloader)
            total_loss_val.append(curr_loss_val)
            total_loss_traj_val.append((batch_loss_traj_val / len(val_dataloader)).item())
            total_loss_label_val.append((batch_loss_label_val / len(val_dataloader)).item())

            if args.step == 1 and curr_loss_val.item() < BEST_LOSS:
                torch.save({
                    'epoch': EPOCH,
                    'model_state_dict': DIFF.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': curr_loss
                }, SAVE_DIR + '/UDiffPlan_state_dict.pth')
                print('saved model')
                BEST_LOSS = curr_loss_val.item()

        if args.step == 1:
            DIFF.train()

        torch.cuda.empty_cache()

        print('loss: ', loss.item(), 'loss traj: ', loss_traj.item(), 'loss label: ', loss_label.item())

        print_metrics(metrics_train)

        print('curr train best ', np.min(total_loss), 'curr val best: ', BEST_LOSS)


        if args.step == 1:
            plt.plot(total_loss, label='train loss')
            plt.plot(total_loss_val, label='val loss')
            plt.plot(total_loss_traj, label='train traj')
            plt.plot(total_loss_traj_val, label='val traj')
            plt.legend()
            plt.savefig(SAVE_DIR+'/train_loss.jpg')
            plt.close()

if __name__ == "__main__":
    main()

import torch
from itertools import product
from pathlib import Path

epochs = [3000, 4000, 5000]
optim_algos = [torch.optim.Adam, torch.optim.RMSprop, torch.optim.SGD]
lrs = [0.01, 0.001, 0.0001]
activation_fns = [torch.nn.functional.relu,
                  torch.nn.functional.sigmoid, torch.nn.functional.tanh]
hidden_layers = [(200, 100, 100), (400, 300, 200),
                 (500, 300, 100)]
# dropout_regs = [0.1, 0.4, 0.7]


def get_all_params():
    all_params = []
    settings = product(epochs, optim_algos, lrs, activation_fns, hidden_layers)
    for i, (epoch, optim, lr, activation, hidden_layer) in enumerate(settings):
        params = {
            'datasource': 'drp_chem',
            'cross_validate': True,
            'verbose': False,
            'train_flag': True,
            'gpu_id': 1,
            'test_data': False,  # TODO: redundant with full_dataset?
            'meta': False,  # TODO: redundant with pretrain?
            'full_dataset': True,
            'fine_tuning': False,
            'pretrain': True,
            'stats_path': Path('./results/cv_statistics.pkl'),
            'cv_stats_overwrite': True,
            'num_hidden_units': hidden_layer,
            'activation_fn': activation,
            'optimizer_fn': optim,
            'k_shot': 10,
            'n_way': 2,
            'inner_lr': lr,
            'meta_lr': lr,
            'pred_lr': 1e-1,
            'meta_batch_size': 10,
            'Lt': 2,
            'Lv': 100,
            'num_inner_updates': 10,
            'kl_reweight': 0.0001,
            'p_dropout_base': 0.4,
            'num_epochs_save': 1000,
            'num_batches': 250,
            'model_name': f'Platipus_{i}',
            'num_epochs': epoch,
            'resume_epoch': 0,
        }
        all_params.append(params)
    return all_params


if __name__ == '__main__':
    all_params = get_all_params()
    print(all_params[0])
    print(len(all_params))

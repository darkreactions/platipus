
from mpi4py import MPI
from hpc_scripts.hpc_params import common_params, meta_params, meta_train, meta_test
from pathlib import Path

from models.meta import main as platipus
from models.meta.platipus_class import Platipus
from models.meta.init_params import init_params
from utils import save_model
import logging
#import models.meta.main as platipus
import sys

nn_configs = [(200, 100, 100), (400, 300, 200), (200, 100), (400, 300)]

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #rank = sys.argv[1]
    print(f'My rank is: {rank}')
    #print(f'My rank is : {rank}', file=sys.stderr)
    #rank = 0

    params = {**common_params, **meta_params}
    params['gpu_id'] = rank
    new_platipus(params)

    params = {**common_params, **meta_params}
    params['gpu_id'] = rank
    old_platipus(params)


def new_platipus(params):
    params['stats_path'] = Path(f'./results/platipus_new_{rank}.pkl')
    params['model_name'] = f'Platipus_new_{rank}'
    params['num_hidden_units'] = nn_configs[rank]

    train_params = {**params, **meta_train}
    params = init_params(train_params)
    logging.basicConfig(filename=Path(save_model(params['model_name'], params))/Path('logfile.log'),
                        level=logging.DEBUG)
    train_params = init_params(train_params)

    for amine in train_params['training_batches']:
        platipus = Platipus(train_params, amine=amine,
                            model_name=train_params['model_name'])
        platipus.meta_train()
        platipus.validate()


def old_platipus(params):
    params['stats_path'] = Path(f'./results/platipus_old_{rank}.pkl')
    params['model_name'] = f'Platipus_old_{rank}'
    params['num_hidden_units'] = nn_configs[rank]
    train_params = {**params, **meta_train}
    train_params = platipus.initialize(
        [train_params['model_name']], train_params)
    platipus.main(train_params)

    test_params = {**params, **meta_test}
    test_params = platipus.initialize([test_params['model_name']], test_params)
    platipus.main(test_params)

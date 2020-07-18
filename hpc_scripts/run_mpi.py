
from mpi4py import MPI
from hpc_scripts.hpc_params import common_params, meta_params, meta_train, meta_test
from pathlib import Path

from models.meta.platipus_class import Platipus
from models.meta.init_params import init_params
from hpc_scripts.param_generator import get_all_params
from utils import save_model
import logging
# import models.meta.main as platipus
import sys
import time


nn_configs = [(200, 100, 100), (400, 300, 200), (200, 100), (400, 300)]
# TAGS
SEND_NEXT_PARAM = 0
PARAM = 1
TERMINATE = 2


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


if __name__ == "__main__":
    node_num = sys.argv[1]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()

    print(f'My rank is: {rank}')
    if rank == 0:
        all_params = get_all_params()
        current_param = 0

        while current_param < len(all_params):
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=SEND_NEXT_PARAM,
                            status=status)
            dest = status.Get_source()
            param = all_params[current_param]
            comm.send(param, dest=dest, tag=PARAM)
            current_param += 1
        process = 0
        while process < 4:
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=SEND_NEXT_PARAM,
                            status=status)
            dest = status.Get_source()
            comm.send(1, dest=dest, tag=TERMINATE)
            process += 1

    else:
        comm.send(1, dest=0, tag=SEND_NEXT_PARAM)
        msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        while tag != TERMINATE:
            print(f'Param on rank {rank}: {msg}')
            time.sleep(1)
            comm.send(1, dest=0, tag=SEND_NEXT_PARAM)
            msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

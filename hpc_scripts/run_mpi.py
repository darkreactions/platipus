
from mpi4py import MPI
from hpc_scripts.hpc_params import common_params, meta_params, meta_train, meta_test
from pathlib import Path
import numpy as np

from models.meta.platipus_class import Platipus
from models.meta.init_params import init_params
from hpc_scripts.param_generator import get_all_params
from utils import save_model
import logging


# TAGS
SEND_NEXT_PARAM = 100
PARAM = 101
TERMINATE = 102


def run_platipus(params):
    logging.basicConfig(filename=Path(save_model(params['model_name'], params))/Path('logfile.log'),
                        level=logging.DEBUG)
    train_params = init_params(params)
    for amine in train_params['training_batches']:
        logging.info(f'Starting process with amine: {amine}')
        platipus = Platipus(train_params, amine=amine,
                            model_name=train_params['model_name'])

        logging.info(f'Begin training with amine: {amine}')
        platipus.meta_train()
        logging.info(f'Begin active learning with amine: {amine}')
        platipus.test_model_actively()
        logging.info(f'Completed active learning with amine: {amine}')


if __name__ == "__main__":
    node_num = 1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()
    all_params = get_all_params()
    print(f'My rank is: {rank}')

    if rank == 0:
        # all_params = get_all_params()
        current_param = 0
        block_len = len(all_params)/3
        start = int(block_len*node_num)
        end = int(block_len*(node_num+1))
        param_block = [i for i in range(start, end)]
        while current_param < len(param_block):
            data = np.empty(1, dtype='i')
            # Waiting for next worker
            comm.Recv(data, source=MPI.ANY_SOURCE,
                      tag=MPI.ANY_TAG, status=status)
            dest = status.Get_source()
            tag = status.Get_tag()
            print(f'Rank {rank} got {data} with tag {tag}')
            param_block_num = param_block[current_param]
            print(
                f'Rank {rank} : Current pointer {current_param} -> {param_block_num}')
            # Sending next job to worker
            data = np.array([param_block_num], dtype='i')
            comm.Send([data, MPI.INT], dest=dest, tag=PARAM)
            current_param += 1
            print(f'Rank {rank} : Incremented Current pointer {current_param}')
        process = 0
        while process < 4:
            data = np.empty(1, dtype='i')
            comm.Recv(data, source=MPI.ANY_SOURCE,
                      tag=MPI.ANY_TAG, status=status)
            dest = status.Get_source()
            data = np.array([0], dtype='i')
            comm.Send([data, MPI.INT], dest=dest, tag=TERMINATE)
            process += 1

    else:
        tag = SEND_NEXT_PARAM
        print(f'Rank {rank} got {data} with tag {tag}')
        while tag != TERMINATE:
            data = np.array([1], dtype='i')
            comm.Send([data, MPI.INT], dest=0, tag=SEND_NEXT_PARAM)
            data = np.empty(1, dtype='i')
            comm.Recv(data, source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == PARAM:
                print(f'Param on rank {rank}: {data} : {all_params[data[0]]} ')
                params = all_params[data[0]]
                params['gpu_id'] = rank-1

                try:
                    run_platipus(params)
                except Exception as e:
                    logging.error(
                        f"In rank {rank} with params: {params} : Exception {e}")
                    continue


from mpi4py import MPI
from hpc_scripts.hpc_params import common_params, meta_params, meta_train, meta_test
from pathlib import Path
import numpy as np

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
SEND_NEXT_PARAM = 100
PARAM = 101
TERMINATE = 102


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
        print(f'Block length = {block_len}')
        start = int(block_len*node_num)
        end = int(block_len*(node_num+1))
        param_block = [i for i in range(start, end)]
        while current_param < len(param_block):
            print(f'Rank {rank} : Waiting for message')
            #msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            data = np.empty(1, dtype='i')
            comm.Recv(data, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            print(f'Rank {rank} : Message received {data}')
            dest = status.Get_source()
            param_block_num = param_block[current_param]
            print(f'Rank {rank}: Selected param_block_num {param_block_num} to send')
            #comm.send(param_block_num, dest=dest, tag=PARAM)
            data = np.array([param_block_num], dtype='i')
            comm.Send([data, MPI.INT], dest=dest, tag=PARAM)
            current_param += 1
        process = 0
        while process < 4:
            data = np.empty(1, dtype='i')
            comm.Recv(data, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            #msg = comm.recv(source=MPI.ANY_SOURCE, tag=SEND_NEXT_PARAM,
            #                status=status)
            dest = status.Get_source()
            #comm.send(1, dest=dest, tag=TERMINATE)
            data = np.array([0], dtype='i')
            comm.Send([data, MPI.INT], dest=dest, tag=TERMINATE)
            process += 1

    else:
        print(f'Rank {rank} sending NEXT PARAM message') 
        data = np.array([1], dtype='i')
        #comm.send(1, dest=0, tag=SEND_NEXT_PARAM)
        comm.Send([data, MPI.INT], dest=0, tag=SEND_NEXT_PARAM)
        data = np.empty(1, dtype='i')
        comm.Recv(data, source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        print(f'Rank {rank} got {data} with tag {tag}')
        while tag != TERMINATE:
            print(f'Param on rank {rank}: {data} : {all_params[data[0]]} ')
            
            time.sleep(1)
            #comm.send(1, dest=0, tag=SEND_NEXT_PARAM)
            comm.Send([data, MPI.INT], dest=0, tag=SEND_NEXT_PARAM)
            #msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            data = np.empty(1, dtype='i')
            comm.Recv(data, source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
    

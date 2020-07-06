from mpi4py import MPI
from hpc_params import common_params, meta_params, meta_train, meta_test
from pathlib import Path
import models.meta.main as platipus
import sys


if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  #rank = sys.argv[1]
  print(f'My rank is: {rank}')
  print(f'My rank is : {rank}', file=sys.stderr)

  params = {**common_params, **meta_params}
  params['gpu_id'] = rank
  params['stats_path'] = Path(f'./results/platipus_{rank}.pkl')
  params['k_shot'] = rank*4
  params['model_name'] = f'Platipus_{rank}'

  train_params = {**params, **meta_train}
  train_params = platipus.initialize([f"PLATIPUS_{rank}"], train_params)
  platipus.main(params)

  test_params = {**params, **meta_test}
  test_params = platipus.initialize([f"PLATIPUS_{rank}"], test_params)
  platipus.main(params)

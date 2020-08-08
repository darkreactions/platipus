import sys
# import time
# from hpc_scripts.hpc_params import common_params, meta_params, meta_train, meta_test
from pathlib import Path

from models.meta.platipus_class import Platipus
from models.meta.init_params import init_params
from hpc_scripts.param_generator import get_all_params
from utils import save_model
import logging


def exp_complete(folder):
    if folder.exists():
        return True
    """
        dirs = [x for x in folder.iterdir() if x.is_dir()]
        if len(dirs) == 16:
    """
    return False


def amine_complete(amine_path):
    """
    if amine_path.exists():
        files_in_amine = [x for x in amine_path.iterdir()]
        # If 1k-5k + cv_stat files exist dont run the amine
        if len(files_in_amine) == 6:
            return True
    """
    return False


def run_platipus(params):
    base_folder = Path(
        f"./results/{params['model_name']}_{params['k_shot']}_shot")
    print(f'Running platipus in base folder {base_folder}')
    # if not base_folder.exists() and params['num_epochs'] == 5000:
    if not exp_complete(base_folder) and params['num_epochs'] == 5000:
        print(f"Found folder running {params['model_name']}")

        logging.basicConfig(filename=Path(save_model(params['model_name'],
                                                     params))/Path('logfile.log'),
                            level=logging.DEBUG)
        train_params = init_params(params)
        for amine in train_params['training_batches']:
            # Checking if amine folder exists, and is incomplete
            amine_path = base_folder / Path(amine)

            if not amine_complete(amine_path):
                logging.info(f'Starting process with amine: {amine}')
                platipus = Platipus(train_params, amine=amine,
                                    model_name=train_params['model_name'],
                                    epoch_al=True)

                logging.info(f'Begin training with amine: {amine}')
                platipus.meta_train()
                logging.info(
                    f'Completed active learning with amine: {amine}')


if __name__ == "__main__":
    exp_num, gpu_id = sys.argv[1:]
    all_params = get_all_params()
    params = all_params[exp_num]
    params['gpu_id'] = gpu_id % 4
    try:
        run_platipus(params)
    except Exception as e:
        logging.error(
            f"In rank {gpu_id} with params: {params} : Exception {e}")

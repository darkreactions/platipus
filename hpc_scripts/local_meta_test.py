from .hpc_params import common_params, local_meta_params, local_meta_train, local_meta_test
from pathlib import Path

from models.meta import main as platipus

params = {**common_params, **local_meta_params}

train_params = {**params, **local_meta_train}
train_params = platipus.initialize(
    [train_params['model_name']], train_params)
platipus.main(train_params)

test_params = {**params, **local_meta_test}
test_params = platipus.initialize([test_params['model_name']], test_params)
platipus.main(test_params)

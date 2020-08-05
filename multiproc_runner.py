from multiprocessing import Pool, Manager


# A function that will take the config and apply it to the model
def f(config):
    result_data, model_config = config
    # Run model here using config and store the results in result_data
    # No need to return anything
    model_result = model_config + ' completed'
    result_data[model_config] = model_result


if __name__ == '__main__':
    # The following dictionary is shared among all processes,
    # so each model will generate its own key based on its config
    manager = Manager()
    result_data = manager.dict()

    # List of arguments where 'abc' is the model configuration
    configs = [(result_data, 'abc'), (result_data, 'def'),
               (result_data, 'ghi'), (result_data, 'jkl'),
               (result_data, 'mno')]

    # Ideally should be atleast c-1 or c-2 where c is the number
    # of logical cores on the machine
    num_processes = 3

    with Pool(num_processes) as pool:
        pool.map(f, configs)

    print(result_data)

from utils.dataset_class import DataSet, Setting
from utils.result_class import Results
from collections import namedtuple, defaultdict
import pickle
from sklearn.neighbors import KNeighborsClassifier
import itertools
from tqdm import tqdm, trange

dataset = pickle.load(open('./data/full_frozen_dataset.pkl', 'rb'))
dataset.print_config()


# categories = ['H', 'Hkx', 'kx']
categories = ['kx']

al_categories = ['ALHk', 'ALk']


all_results = []
model_name = 'KNN3'

# Set all possible combinations
combinations = []
ft_params = {
    'n_neighbors': [i for i in range(1, 10)],
    'leaf_size': [i for i in range(1, 51)],
    'p': [i for i in range(1, 4)]
}
keys, values = zip(*ft_params.items())
for bundle in itertools.product(*values):
    combinations.append(dict(zip(keys, bundle)))
# print(combinations)


for ct in trange(len(categories), desc='Category'):
    cat = categories[ct]
    selection = 'random'
    for c in trange(len(combinations), desc='Combination'):
        combo = combinations[c]
        result = Results(al=False, category=cat, total_sets=dataset.num_draws,
                         model_name=model_name, model_setting=combo,
                         selection=selection)
        for set_id in trange(dataset.num_draws, desc='Set'):
            data = dataset.get_dataset(cat, set_id, selection)
            knn = KNeighborsClassifier(**combo)

            amines = list(data.keys())
            for a in range(len(amines)):
                amine = amines[a]
                d = data[amine]
                knn.fit(d['x_t'], d['y_t'])
                y_pred = knn.predict(d['x_v'])
                result.add_result(d['y_v'], y_pred, set_id, amine)
            all_results.append(result)

        pickle.dump(all_results, open('./results/knn_results.pkl', 'wb'))

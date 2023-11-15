from algorithms.hasuraRequest import fetch_algorithm, fetch_real_dataset, fetch_simulator_dataset
from algorithms.isolationForest import isolation_forest

failure_types = []

def select_algorithm(algorithm_settings):
    algorithm_settings_id = algorithm_settings.get('id')

    fetched_algorithm = fetch_algorithm(algorithm_settings.get('algorithm_id')).get('data').get('algorithms_by_pk').get('algorithmName')

    fetched_data = [];
    
    if algorithm_settings.get('simulator_id') != None:
        fetched_data = fetch_simulator_dataset(algorithm_settings.get('simulator_id')).get('data').get('datasets')[0].get('result')
    else:
        fetched_data =fetch_real_dataset(algorithm_settings.get('real_dataset_id')).get('data').get('datasets')[0].get('result')

    if (fetched_algorithm == 'IsolationForest'):
        return isolation_forest(algorithm_settings_id, fetched_data)

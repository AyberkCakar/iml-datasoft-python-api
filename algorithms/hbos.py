from pyod.models.hbos import HBOS
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def hbos_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data.drop(['tag', 'time'], axis=1))

    hbos = HBOS()
    hbos.fit(features)
    model_predictions = hbos.labels_

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

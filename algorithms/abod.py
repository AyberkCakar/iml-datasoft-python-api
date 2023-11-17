from pyod.models.abod import ABOD
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def abod_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    abod = ABOD()
    abod.fit(features)
    model_predictions = abod.labels_

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

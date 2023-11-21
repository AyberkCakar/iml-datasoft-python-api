from pyod.models.cblof import CBLOF
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def cblof_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    cblof = CBLOF(n_clusters=20, contamination=0.1)
    cblof.fit(features)
    model_predictions = cblof.labels_

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

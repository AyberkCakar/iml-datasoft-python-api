from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def lof_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)

    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    predictions = lof.fit_predict(features)
    model_predictions = [1 if x == -1 else 0 for x in predictions]

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

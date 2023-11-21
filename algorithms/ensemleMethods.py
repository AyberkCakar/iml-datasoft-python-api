import numpy as np
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def ensemble_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    models = [KNN(), LOF(), ABOD()]
    model_predictions = [model.fit_predict(features) for model in models]

    averaged_predictions = np.mean(model_predictions, axis=0)
    binary_predictions = np.round(averaged_predictions)

    return calculate_metrics_and_update_algorithm_result(true_labels, binary_predictions, algorithm_settings_id, algorithm_id)

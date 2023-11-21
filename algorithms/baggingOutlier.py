from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import numpy as np
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def feature_bagging_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit_predict(features)
    lof_scores = -lof.negative_outlier_factor_

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(features)
    distances, _ = knn.kneighbors(features)
    knn_scores = np.mean(distances, axis=1)

    average_score = (lof_scores + knn_scores) / 2

    threshold = np.percentile(average_score, 95)
    model_predictions = (average_score > threshold).astype(int)

    result = {
        # 'average_score': average_score.tolist(),
        # 'is_outlier': model_predictions.tolist()
    }

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

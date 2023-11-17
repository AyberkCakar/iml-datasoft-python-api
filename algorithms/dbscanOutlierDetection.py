from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def dbscan_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(scaled_features)

    model_predictions = [1 if x == -1 else 0 for x in dbscan.labels_]

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

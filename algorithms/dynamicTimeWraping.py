from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def dtw_clustering(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    model = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=2)
    labels = model.fit_predict(features)

    return calculate_metrics_and_update_algorithm_result(true_labels, labels, algorithm_settings_id, algorithm_id)

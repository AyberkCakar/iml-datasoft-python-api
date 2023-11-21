import rrcf
import pandas as pd
import numpy as np
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def robust_random_cut_forest(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = data[sensor_types]

    num_trees = 10
    tree_size = 64
    forest = [rrcf.RCTree() for _ in range(num_trees)]

    for index, point in features.iterrows():
        for tree in forest:
            tree.insert_point(point.values, index=index)
            if len(tree.leaves) > tree_size:
                tree.forget_point(index - tree_size)

    anomaly_scores = {index: np.mean(
        [tree.codisp(index) for tree in forest]) for index in range(len(features))}

    threshold = np.percentile(list(anomaly_scores.values()), 95)
    model_predictions = [1 if anomaly_scores[i] >
                         threshold else 0 for i in range(len(features))]

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

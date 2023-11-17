from sklearn.ensemble import IsolationForest
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def isolation_forest(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)

    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.1, random_state=42)
    iso_forest.fit(features)

    predictions = iso_forest.predict(features)
    model_predictions = [1 if x == -1 else 0 for x in predictions]

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

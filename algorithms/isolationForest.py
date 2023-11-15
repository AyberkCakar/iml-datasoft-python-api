from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result

def isolation_forest(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)

    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = data.drop(['tag', 'time'], axis=1)

    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso_forest.fit(features)

    predictions = iso_forest.predict(features)
    model_predictions = [1 if x == -1 else 0 for x in predictions]
    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions)
    recall = recall_score(true_labels, model_predictions)
    f1 = f1_score(true_labels, model_predictions)

    data = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    return set_algorithm_result(data, algorithm_settings_id)
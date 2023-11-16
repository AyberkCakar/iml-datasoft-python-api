from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def lof_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)

    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    predictions = lof.fit_predict(features)
    model_predictions = [1 if x == -1 else 0 for x in predictions]

    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions)
    recall = recall_score(true_labels, model_predictions)
    f1 = f1_score(true_labels, model_predictions)

    data = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    return set_algorithm_result(data, algorithm_settings_id)
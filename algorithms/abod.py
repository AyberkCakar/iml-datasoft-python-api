from pyod.models.abod import ABOD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def abod_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    # ABOD modeli
    abod = ABOD()
    abod.fit(features)
    model_predictions = abod.labels_
   
    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions)
    recall = recall_score(true_labels, model_predictions)
    f1 = f1_score(true_labels, model_predictions)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


    return set_algorithm_result(result, algorithm_settings_id)

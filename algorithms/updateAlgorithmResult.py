from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from algorithms.hasuraRequest import update_algorithm_result


def calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id):
    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(
        true_labels, model_predictions, zero_division=0)
    recall = recall_score(true_labels, model_predictions, zero_division=0)
    f1 = f1_score(true_labels, model_predictions, zero_division=0)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    return update_algorithm_result(result, algorithm_settings_id, algorithm_id)

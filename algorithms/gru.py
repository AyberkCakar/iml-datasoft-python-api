from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def gru_anomaly_detection(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = data.drop(['tag', 'time'], axis=1)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = scaled_features.reshape(
        scaled_features.shape[0], scaled_features.shape[1], 1)

    model = Sequential([
        GRU(units=50, return_sequences=True,
            input_shape=(scaled_features.shape[1], 1)),
        GRU(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(scaled_features, scaled_features,
              epochs=2, batch_size=32, shuffle=True)

    predictions = model.predict(scaled_features)
    if predictions.shape[1] == 1:
        predictions = np.expand_dims(predictions, axis=-1)

    mse = np.mean(np.power(scaled_features - predictions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    model_predictions = (mse > threshold).astype(int)

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def lstm_anomaly_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = data[sensor_types]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = Sequential([
        LSTM(units=50, return_sequences=True,
             input_shape=(scaled_features.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    reshaped_features = np.reshape(
        scaled_features, (scaled_features.shape[0], scaled_features.shape[1], 1))
    model.fit(reshaped_features, scaled_features,
              epochs=20, batch_size=32, shuffle=True)

    predictions = model.predict(reshaped_features)
    mse = np.mean(np.power(scaled_features - predictions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    model_predictions = (mse > threshold).astype(int)

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

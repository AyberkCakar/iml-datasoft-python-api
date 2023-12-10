import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def autoencoder_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    input_layer = Input(shape=(X_scaled.shape[1],))
    encoded = Dense(16, activation='relu')(input_layer)
    decoded = Dense(X_scaled.shape[1], activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=32,
                    shuffle=True, validation_split=0.2, verbose=0)

    predictions = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    model_predictions = (mse > threshold).astype(int)

    result = {
        # 'mse': mse.tolist(),
        # 'is_outlier': model_predictions.tolist()
    }

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

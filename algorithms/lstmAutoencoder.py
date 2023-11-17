import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.optimizers import Adam
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def build_lstm_autoencoder(timesteps, num_features, latent_dim=64):
    # Encoder
    inputs = Input(shape=(timesteps, num_features))
    encoded = LSTM(latent_dim, activation='relu')(inputs)

    # Decoder
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(num_features, return_sequences=True,
                   activation='relu')(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                        loss='mean_squared_error')

    return autoencoder


def lstm_autoencoder(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    timesteps = 10
    samples = len(features) // timesteps
    num_features = features.shape[1]  # 3 features

    features_reshaped = features.values[:samples *
                                        timesteps].reshape((samples, timesteps, num_features))

    autoencoder = build_lstm_autoencoder(timesteps, num_features)
    autoencoder.fit(features_reshaped, features_reshaped,
                    epochs=20, batch_size=256, shuffle=True)

    reconstructed = autoencoder.predict(features_reshaped)
    reconstruction_error = np.mean(
        np.power(features_reshaped - reconstructed, 2), axis=1)

    threshold = np.quantile(reconstruction_error, 0.95)
    predictions = reconstruction_error > threshold
    model_predictions = np.where(predictions, 1, 0)

    downsampled_true_labels = true_labels[::10]
    downsampled_true_labels = downsampled_true_labels.astype(
        int)  # Tamsayıya dönüştür

    binary_predictions = np.any(model_predictions == 1, axis=1).astype(int)

    return calculate_metrics_and_update_algorithm_result(downsampled_true_labels, binary_predictions, algorithm_settings_id, algorithm_id)

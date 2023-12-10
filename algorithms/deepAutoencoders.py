from keras.models import Model
from keras.layers import Input, Dense
import pandas as pd
import numpy as np
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def deep_autoencoder_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    # Autoencoder modeli
    input_layer = Input(shape=(features.shape[1],))
    encoded = Dense(100, activation='relu')(input_layer)
    encoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(features.shape[1], activation='sigmoid')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(features, features, epochs=20, batch_size=32,
                    shuffle=True, validation_split=0.3, verbose=0)

    # Yeniden yapılandırma hatası
    predictions = autoencoder.predict(features)
    mse = np.mean(np.power(features - predictions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    model_predictions = (mse > threshold).astype(int)

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)

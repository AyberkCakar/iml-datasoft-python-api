import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def build_lstm_autoencoder(timesteps, num_features, latent_dim=64):
    # Encoder
    inputs = Input(shape=(timesteps, num_features))
    encoded = LSTM(latent_dim, activation='relu')(inputs)

    # Decoder
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(num_features, return_sequences=True, activation='relu')(decoded)

    # Otomatik kodlayıcı modeli
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return autoencoder

def lstm_autoencoder(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    # Örnek olarak, her bir örnek 10 zaman adımından oluşsun
    timesteps = 10  
    samples = len(features) // timesteps
    num_features = features.shape[1]  # 3 özellik

    # Veri setini yeniden şekillendir
    features_reshaped = features.values[:samples*timesteps].reshape((samples, timesteps, num_features))


    # Modeli oluştur ve eğit
    autoencoder = build_lstm_autoencoder(timesteps, num_features)
    autoencoder.fit(features_reshaped, features_reshaped, epochs=20, batch_size=256, shuffle=True)

    # Yeniden oluşturma hatasını hesapla
    reconstructed = autoencoder.predict(features_reshaped)
    reconstruction_error = np.mean(np.power(features_reshaped - reconstructed, 2), axis=1)
    
    # Anomali tespiti için bir eşik değeri belirle
    # Eşik değeri belirleme yöntemi veriye bağlı olarak değişebilir
    threshold = np.quantile(reconstruction_error, 0.95)
    predictions = reconstruction_error > threshold
    model_predictions = np.where(predictions, 1, 0)

    downsampled_true_labels = true_labels[::10]
    downsampled_true_labels = downsampled_true_labels.astype(int)  # Tamsayıya dönüştür

    binary_predictions = np.any(model_predictions == 1, axis=1).astype(int)

    # Performans metriklerini hesapla
    accuracy = accuracy_score(downsampled_true_labels, binary_predictions)
    precision = precision_score(downsampled_true_labels, binary_predictions)
    recall = recall_score(downsampled_true_labels, binary_predictions)
    f1 = f1_score(downsampled_true_labels, binary_predictions)

    # Performans metriklerini bir sözlükte topla
    performance_data = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    return set_algorithm_result(performance_data, algorithm_settings_id)
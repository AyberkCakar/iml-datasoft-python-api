import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalization_data(data, sensor_types):
    features = data[sensor_types]
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return pd.DataFrame(normalized_features, columns=[sensor_types])

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
import numpy as np

def normalize_data(data, method=0):
    """Chuẩn hóa dữ liệu."""
    if method == 1:
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == 2:
        scaler = StandardScaler()
    elif method == 3:
        scaler = RobustScaler()
    else:
        print("Không chuẩn hóa dữ liệu!")
        return data, None

    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return data_scaled, scaler

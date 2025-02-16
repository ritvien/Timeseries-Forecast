import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _boxplot(data, scale_factor=1.5):
    """
    Phát hiện và vẽ boxplot để kiểm tra outliers trong dữ liệu.
    
    Parameters:
        data (pd.Series or list): Dữ liệu đầu vào
        
    Returns:
        outliers (list): Danh sách outliers
    """
    clean_data = data.dropna().values
    Q1 = np.percentile(clean_data, 25)
    Q3 = np.percentile(clean_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - scale_factor * IQR
    upper_bound = Q3 + scale_factor * IQR
    
    outlier_indices = (data < lower_bound) | (data > upper_bound)
    
    outliers = pd.DataFrame({'Value': data[outlier_indices]})

    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Series", color='blue', alpha=0.7)
    
    plt.scatter(outliers.index, outliers['Value'], color='red', label="Outliers", zorder=3)
    
    plt.axhline(y=lower_bound, color='green', linestyle='--', label=f"Lower: {lower_bound}")
    plt.axhline(y=upper_bound, color='orange', linestyle='--', label=f"Upper: {upper_bound}")
    
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend(ncol=2)
    plt.show()
    
    print(f"Phát hiện {len(outliers)} outliers.")
    return outliers

def _rolling(data, window_size=30, threshold=3):
    """
    Phát hiện bất thường trong chuỗi thời gian bằng rolling mean và rolling std.

    Parameters:
        data (pd.Series): Chuỗi dữ liệu gốc.
        window_size (int): Kích thước cửa sổ trượt.
        threshold (float): Ngưỡng phát hiện outliers (số lần độ lệch chuẩn).

    Returns:
        outliers (pd.DataFrame): DataFrame chứa index và giá trị của outliers.
    """
    rolling_mean = data.rolling(window=window_size, center=True).mean()
    rolling_std = data.rolling(window=window_size, center=True).std()

    # Xác định ngưỡng trên và dưới
    upper_bound = rolling_mean + threshold * rolling_std
    lower_bound = rolling_mean - threshold * rolling_std

    # Xác định các điểm bất thường
    outlier_indices = (data > upper_bound) | (data < lower_bound)
    outliers = data[outlier_indices]

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Original Data", color='blue', alpha=0.7)
    plt.plot(rolling_mean, label=f"Rolling Mean (window={window_size})", color='green', linestyle="--")
    plt.fill_between(data.index, lower_bound, upper_bound, color='gray', alpha=0.3, label="Threshold Range")
    plt.scatter(outliers.index, outliers, color='red', label="Outliers", zorder=3)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Anomaly Detection using Rolling Mean & Std (window={window_size}, threshold={threshold})")
    plt.legend(ncol=2)
    plt.show()

    print(f"Phát hiện {len(outliers)} outliers.")
    return outliers.to_frame(name="Value")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, r2_score

PATH_IMG = "../tests/image/"

def inverse_scale(data_scaled, scaler):
    """
    Đảo ngược quá trình chuẩn hóa dữ liệu.

    Parameters:
        data_scaled (np.array): Dữ liệu đã được chuẩn hóa.
        scaler: Đối tượng scaler đã sử dụng để chuẩn hóa dữ liệu.

    Returns:
        np.array: Dữ liệu gốc sau khi đảo ngược chuẩn hóa.
    """
    if scaler is None:
        return data_scaled  
    data_original = scaler.inverse_transform(np.array(data_scaled).reshape(-1, 1))
    return data_original.flatten()


def plot_forecast(train_data, y_test, y_pred, model, index, conf_int=None):
    """
    Vẽ và lưu biểu đồ dự báo.

    Args:
        train_data (pd.Series): Dữ liệu huấn luyện.
        y_test (pd.Series): Dữ liệu thực tế.
        y_pred (np.array): Giá trị dự báo.
        model: Đối tượng model đã được huấn luyện (có thuộc tính `__class__.__name__`).
        index (int): Index của dòng metrics đã lưu.
        conf_int (np.array, optional): Khoảng tin cậy của dự báo.
    """
    plt.figure(figsize=(12, 6))

    train_index = train_data.index
    test_index = y_test.index

    model_name = model.model.__class__.__name__

    plt.plot(train_index, train_data, label="Original Data", color='gray', alpha=0.5)
    plt.plot(test_index, y_test, label="Actual", color='blue')
    plt.plot(test_index[:len(y_pred)], y_pred, label="Predicted", color='red', linestyle='dashed')

    # Vẽ khoảng tin cậy nếu có
    if conf_int is not None:
        conf_int = np.array(conf_int).reshape(-1, 2)
        lower_bound, upper_bound = conf_int[:, 0], conf_int[:, 1]
        plt.fill_between(test_index[:len(y_pred)], lower_bound, upper_bound, 
                         color='pink', alpha=0.3, label="Confidence Interval")

    # Cài đặt đồ thị
    plt.title(f"{model_name} Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(ncol=4)

    # Lưu hình ảnh với tên index_{Model}.png
    file_path = f"{PATH_IMG}_{index}_{model_name}.png"
    plt.savefig(file_path, dpi=300)
    plt.show()

    plt.close()

    print(f"Plot saved to {file_path}")




def compute_metrics(model, scaler, y_test, y_pred):
    """
    Tính toán các chỉ số đánh giá mô hình và trả về dưới dạng một dòng DataFrame.

    Args:
        model (object): Đối tượng mô hình (VD: ARIMA, RandomForest, LSTM,...)
        scaler (object): Bộ scaler đã sử dụng (VD: MinMaxScaler, StandardScaler,...)
        y_test (array-like): Giá trị thực tế
        y_pred (array-like): Giá trị dự đoán

    Returns:
        pd.DataFrame: Một dòng DataFrame chứa metrics đánh giá mô hình
    """

    model_name = model.model.__class__.__name__

    try:
        model_params = model.get_params()
    except AttributeError:
        try:
            model_params = model.order  # Cho ARIMA
        except AttributeError:
            model_params = "Unknown"

    scaler_name = scaler.__class__.__name__
    execution_date = datetime.today().strftime("%d-%m-%Y")
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = r2_score(y_test, y_pred)

    result = pd.DataFrame([{
        "Model": model_name,
        "Scaler": scaler_name,
        "Params": str(model_params),
        "MAPE": mape,
        "RMSE": rmse,
        "R2 Score": r2,
        "Execution Date": execution_date
    }])

    return result



def save_metrics_to_csv(df, file_path):
    """
    Lưu DataFrame metrics vào file CSV. Nếu file chưa tồn tại thì tạo mới, 
    nếu đã tồn tại thì append dữ liệu vào.

    Args:
        df (pd.DataFrame): DataFrame chứa metrics của mô hình.
        file_path (str): Đường dẫn đến file CSV cần lưu.

    Returns:
        int: Index của dòng vừa được lưu.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Đảm bảo thư mục tồn tại

    file_exists = os.path.isfile(file_path)

    if file_exists:
        existing_df = pd.read_csv(file_path)
        last_index = existing_df.index[-1] + 1 if not existing_df.empty else 0
    else:
        last_index = 0

    df.index = range(last_index, last_index + len(df))

    # Lưu vào CSV
    df.to_csv(file_path, mode="a", index=False, header=not file_exists, encoding="utf-8")

    print(f"Metrics saved to {file_path}")

    return last_index 

from pmdarima import auto_arima
import numpy as np

class SARIMAModel:
    def __init__(self, train_data, p=None, d=None, q=None, P=None, D=None, Q=None, m=1):
        """
        Khởi tạo mô hình SARIMA.

        :param train_data: Dữ liệu huấn luyện (chuỗi thời gian).
        """
        self.train_data = train_data
        self.model = None
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m

    def fit(self):
        """
        Huấn luyện mô hình SARIMA với các tham số chỉ định hoặc tìm tham số tối ưu.

        Args:
            p, d, q (int, optional): Tham số của phần ARIMA.
            P, D, Q (int, optional): Tham số của phần mùa vụ.
            m (int, optional): Chu kỳ mùa vụ (mặc định = 1, tức là không có mùa vụ).
        """
        self.model = auto_arima(
            self.train_data,
            p=self.p if self.p is not None else 0, max_p=self.p if self.p is not None else 5,
            d=self.d if self.d is not None else 0, max_d=self.d if self.d is not None else 2,
            q=self.q if self.q is not None else 0, max_q=self.q if self.q is not None else 5,
            P=self.P if self.P is not None else 0, max_P=self.P if self.P is not None else 2,
            D=self.D if self.D is not None else 0, max_D=self.D if self.D is not None else 1,
            Q=self.Q if self.Q is not None else 0, max_Q=self.Q if self.Q is not None else 2,
            m=self.m, 
            seasonal=(self.m > 1),  
            stepwise=True,
            trace=True
        )

        print(f"Best params: {self.model.summary()}")

    def predict(self, length):
        """
        Dự báo nhiều bước phía trước.

        :param length: Số bước cần dự báo.
        :return: DataFrame chứa giá trị dự báo và khoảng tin cậy.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call fit() before predicting.")

        forecast, conf_int = self.model.predict(n_periods=length, return_conf_int=True)

        return np.array(forecast), np.array(conf_int)

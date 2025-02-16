from pmdarima import auto_arima
import numpy as np

class ARIMAModel:
    def __init__(self, train_data, p=None, d=None, q=None):
        """
        Khởi tạo mô hình ARIMA.

        :param train_data: Dữ liệu huấn luyện (chuỗi thời gian).
        """
        self.train_data = train_data
        self.model = None
        self.p = p
        self.d = d
        self.q = q

    def fit(self):
        """
        Huấn luyện mô hình ARIMA với các tham số chỉ định hoặc tìm tham số tối ưu.

        Args:
            p (int, optional): Giá trị của tham số p (số lượng bậc trễ của phần AR).
            d (int, optional): Giá trị của tham số d (số lần sai phân để làm dừng dữ liệu).
            q (int, optional): Giá trị của tham số q (số lượng bậc trễ của phần MA).

        Những tham số nào được truyền vào sẽ cố định, các tham số còn lại sẽ được tối ưu tự động.
        """
        self.model = auto_arima(
            self.train_data,
            p=self.p if self.p is not None else 0, max_p=self.p if self.p is not None else 5,
            d=self.d if self.d is not None else 0, max_d=self.d if self.d is not None else 2,
            q=self.q if self.q is not None else 0, max_q=self.q if self.q is not None else 5,
            stepwise=True,
            seasonal=False,
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

from statsmodels.tsa.forecasting.theta import ThetaModel

class THETAModel:
    def __init__(self, train_data, periods):
        """
        Khởi tạo mô hình Theta.
        
        Args:
            train_data (pd.Series): Dữ liệu huấn luyện (chuỗi thời gian).
        """
        self.train_data = train_data
        self.periods = periods
        self.model = None

    def fit(self):
        """
        Huấn luyện mô hình Theta với chu kỳ mùa vụ được chỉ định.

        Args:
            periods (int): Độ dài của chu kỳ mùa vụ.
        """

        self.model = ThetaModel(self.train_data, period=self.periods).fit()
        print(f"Theta Model trained successfully with period={self.periods}.")

    def predict(self, length):
        """
        Dự báo nhiều bước phía trước.

        Args:
            length (int): Số bước cần dự báo.

        Returns:
            np.array: Giá trị dự báo.
        """
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện. Vui lòng gọi fit() trước.")

        forecast = self.model.forecast(steps=length)
        return forecast.to_numpy(), None
from model_module.arima import ARIMAModel
from model_module.sarima import SARIMAModel
from model_module.theta import THETAModel

class Model:
    def __init__(self, train_data):
        self.train_data = train_data
        self.model = None
        self.predictions = None
        self.conf_int = None
    def run(self):
        print("Lựa chọn model thống kê:")
        print("1. ARIMA\n2. SARIMA\n3. THETA (Mặc định)")
        model_type = int(input())
        if model_type == 1:
            print("Nhập hyperparameters (p,d,q):")
            print("1. Nhập\n2. Tự tối ưu (mặc định)")
            option = int(input())
            if option == 1:
                try:
                    p_input = input("Nhập p (Số lượng lag trong thành phần AR) hoặc 'auto' để tối ưu: ")
                    d_input = input("Nhập d (Số lần lấy sai phân để chuỗi dừng) hoặc 'auto' để tối ưu: ")
                    q_input = input("Nhập q (Số lượng lag trong thành phần MA) hoặc 'auto' để tối ưu: ")
                    
                    # Nếu người dùng nhập 'auto', thì đặt tham số tương ứng thành None để tối ưu tự động
                    p = int(p_input) if p_input != 'auto' else None
                    d = int(d_input) if d_input != 'auto' else None
                    q = int(q_input) if q_input != 'auto' else None
                except:
                    print("Nhập không hợp lệ, chọn tham số mặc định")
                    p,d,q = None, None, None
            else:
                p,d,q = None, None, None

            self.model = ARIMAModel(self.train_data,p, d, q)
        elif model_type == 2:
            print("Nhập hyperparameters (p,d,q, P, D, Q, m):")
            print("1. Nhập\n2. Tự tối ưu (mặc định)")
            option = int(input())
            if option == 1:
                try:
                    # Nhập các tham số và cho phép nhập 'auto' để tối ưu
                    p_input = input("Nhập p (Số lượng lag trong thành phần AR) hoặc 'auto' để tối ưu: ")
                    d_input = input("Nhập d (Số lần lấy sai phân để chuỗi dừng) hoặc 'auto' để tối ưu: ")
                    q_input = input("Nhập q (Số lượng lag trong thành phần MA) hoặc 'auto' để tối ưu: ")
                    
                    # Nhập các tham số mùa vụ (seasonal)
                    P_input = input("Nhập P (Số lượng lag trong thành phần AR mùa vụ) hoặc 'auto' để tối ưu: ")
                    D_input = input("Nhập D (Số lần lấy sai phân mùa vụ) hoặc 'auto' để tối ưu: ")
                    Q_input = input("Nhập Q (Số lượng lag trong thành phần MA mùa vụ) hoặc 'auto' để tối ưu: ")
                    m_input = input("Nhập m (Chu kỳ mùa vụ) hoặc 'auto' để tối ưu: ")

                    # Kiểm tra và chuyển đổi các tham số
                    p = int(p_input) if p_input != 'auto' else None
                    d = int(d_input) if d_input != 'auto' else None
                    q = int(q_input) if q_input != 'auto' else None
                    P = int(P_input) if P_input != 'auto' else None
                    D = int(D_input) if D_input != 'auto' else None
                    Q = int(Q_input) if Q_input != 'auto' else None
                    m = int(m_input) if m_input != 'auto' else 1

                except ValueError:
                    print("Nhập không hợp lệ, chọn tham số mặc định")
                    p, d, q, P, D, Q, m = None, None, None, None, None, None, 1

            else:
                p, d, q, P, D, Q, m = None, None, None, None, None, None, 1
            self.model = SARIMAModel(self.train_data, p, d, q, P, D, Q, m)
        else:
            print("Lựa chọn THETA model")
            while True:
                try:
                    print("Nhập mùa vụ:")
                    period = int(input())
                    self.model = THETAModel(self.train_data, period)
                    break

                except Exception as e:
                    print(e)
                    print("Nhập không hợp lệ, nhập lại")
                    continue
        print("Hoàn tất khởi tạo model")

    def fit(self):
        print("#--------------Đang tối ưu, train model----------------#")
        self.model.fit()
    
    def predict(self, length):
        print("#---------------Predict----------------#")
        self.predictions, self.conf_int = self.model.predict(length)

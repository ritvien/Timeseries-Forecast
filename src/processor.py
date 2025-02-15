import pandas as pd
import numpy as np
import anomaly_detection as ad

class Processor:
    def __init__(self, train_data):
        self.train_data = train_data
    
    def run(self):
        self.preprocess()

        while True:
            print("#----------Phát hiện bất thường-----------#")
            print("0. Đóng chương trình\n1. Không dùng\n2. Boxplot\n3. Rolling window")
            self.outlier_type = int(input())
            if self.outlier_type == 1:
                anomalies = None
            elif self.outlier_type == 2:
                print("Điều chỉnh mức độ nhạy (Hệ số nhân với IQR):")
                print("1. 1.5 (phổ biến, mặc định)\n2. 1.0 (nhạy cảm)\n3. 3.0 (ít nhạy))")
                factor = int(input())
                if factor == 1:
                    scale_factor = 1.5
                elif factor == 2:
                    scale_factor = 1
                elif factor == 3:
                    scale_factor = 3
                else:
                    print("Lựa chọn không hợp lệ, lấy giá trị mặc định: 1.5")
                    outliers = ad._boxplot(self.train_data['temperatures'])
                outliers = ad._boxplot(self.train_data['temperatures'], scale_factor)
                return True
            elif self.outlier_type == 3:
                try:
                    window_size = int(input("Nhập cỡ cửa sổ trượt(int): "))
                    factor = float(input("Nhập hệ số nhạy (mean ± factor*std) - mặc định là 3:"))
                    outliers = ad._rolling(self.train_data['temperatures'], window_size, factor)
                except:
                    print("Nhập số không hợp lệ, lấy giá trị mặc định: (30,3)")
                    outliers = ad._rolling(self.train_data['temperatures'])
                finally:
                    return True
            elif self.outlier_type == 0:
                return False
            else:
                print("Lựa chọn không hợp lệ")
                continue      
    def preprocess(self):
        '''
            Xử lý date
        '''
        self.train_data["Date"] = pd.to_datetime(self.train_data["Date"])
        self.train_data.set_index("Date", inplace=True)
        self.train_data = self.train_data.asfreq("D")
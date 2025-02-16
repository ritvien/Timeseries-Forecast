import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anomaly_detection as ad

class Processor:
    def __init__(self, train_data):
        self.train_data = train_data
    
    def run(self):
        '''
            Chạy quá trình xử lý dữ liệu
        '''
        self.preprocess()

        while True:
            print("#----------Phát hiện bất thường-----------#")
            print("0. Đóng chương trình\n1. Không dùng\n2. Boxplot\n3. Rolling window")
            self.outlier_type = int(input())
            if self.outlier_type == 1:
                outliers = None
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
 
            elif self.outlier_type == 3:
                try:
                    window_size = int(input("Nhập cỡ cửa sổ trượt(int): "))
                    factor = float(input("Nhập hệ số nhạy (mean ± factor*std) - mặc định là 3:"))
                    outliers = ad._rolling(self.train_data['temperatures'], window_size, factor)
                except:
                    print("Nhập số không hợp lệ, lấy giá trị mặc định: (30,3)")
                    outliers = ad._rolling(self.train_data['temperatures'])

            elif self.outlier_type == 0:
                return False
            
            else:
                print("Lựa chọn không hợp lệ")
                continue

            print("#----------Thay thế outliers và điền giá trị thiếu-----------#")
            print("Lựa chọn phương pháp điền giá trị thiếu:")
            print("1. mean\n2. backward\n3. forward\n. 4. interpolate(mặc định)")
            fill_method = int(input())
            if fill_method == 1:
                method = "mean"
            elif fill_method == 2:
                method = "backward"
            elif fill_method == 3:
                method = "forward"
            elif fill_method == 4:
                method = "interpolate"
            else:
                print("Lựa chọn không hợp lệ, fill mặc định là interpolate")
                method = "interpolate"

            self.train_filled = self.replace_and_fill_outliers(outliers, method)
            

            print("#--------------Chuẩn hóa dữ liệu------------#")
            









    def preprocess(self):
        '''
            Xử lý date
        '''
        self.train_data["Date"] = pd.to_datetime(self.train_data["Date"])
        self.train_data.set_index("Date", inplace=True)
        self.train_data = self.train_data.asfreq("D")

    def replace_and_fill_outliers(self, outliers, method="interpolate"):
        """
        Thay thế outliers trong Series bằng NaN và điền lại bằng các phương pháp khác nhau.

        Parameters:
            outliers (pd.DataFrame): DataFrame chứa outliers (index có thể khác `data.index`).
            method (str): Phương pháp điền giá trị ['mean', 'backward', 'forward', 'interpolate'].
        """
        valid_outliers_idx = outliers.index.intersection(self.train_data.index)

        cleaned_data = self.train_data.copy()

        cleaned_data.loc[valid_outliers_idx] = np.nan

        if method == "mean":
            filled_data = cleaned_data.fillna(cleaned_data.mean())
        elif method == "backward":
            filled_data = cleaned_data.fillna(method="bfill")
        elif method == "forward":
            filled_data = cleaned_data.fillna(method="ffill")
        elif method == "interpolate":
            filled_data = cleaned_data.interpolate(method="linear")  
        else:
            print("Lựa chọn không hợp lệ, chọn mặc định fill: interpolate")
            filled_data = cleaned_data.interpolate(method="linear")
        # Vẽ biểu đồ so sánh trước và sau khi xử lý
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_data, label="Original Data", color='blue', alpha=0.6)
        plt.plot(filled_data, label="Cleaned Data", color='green', linestyle="--")
        plt.scatter(valid_outliers_idx, self.train_data.loc[valid_outliers_idx], color='red', label="Replaced Outliers", zorder=3)

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Outlier Replacement using {method} method")
        plt.legend()
        plt.show()

        print(f"Thay thế {len(valid_outliers_idx)} outliers và fill: '{method}'.")
        return filled_data
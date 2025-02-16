import pandas as pd
import process_module.anomaly_detection as ad
import process_module.scaler as scaler
import process_module.replace_and_fill_outliers as fill
class Processor:
    def __init__(self, train_data):
        self.train_data = train_data
        self.scaler = None
        self.train_filled = None
        self.train_scaled = None
        self.outliers = None
    
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
                self.outliers = None
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
                    scale_factor = 1.5
                self.outliers = ad._boxplot(self.train_data['temperatures'], scale_factor)
 
            elif self.outlier_type == 3:
                try:
                    window_size = int(input("Nhập cỡ cửa sổ trượt(int): "))
                    factor = float(input("Nhập hệ số nhạy (mean ± factor*std) - mặc định là 3:"))
                    self.outliers = ad._rolling(self.train_data['temperatures'], window_size, factor)
                except:
                    print("Nhập số không hợp lệ, lấy giá trị mặc định: (30,3)")
                    self.outliers = ad._rolling(self.train_data['temperatures'])

            elif self.outlier_type == 0:
                return False
            
            else:
                print("Lựa chọn không hợp lệ")
                continue
            print("#----------Thay thế outliers và điền giá trị thiếu-----------#")
            print("Lựa chọn phương pháp điền giá trị thiếu:")
            print("1. mean\n2. backward\n3. forward\n4. interpolate(mặc định)")

            self.train_filled = fill.replace_and_fill_outliers(self.train_data,self.outliers)
            

            print("#--------------Chuẩn hóa dữ liệu------------#")
            print("Lựa chọn phương pháp chuẩn hóa:")
            print("1. Standard\n2. Minmax\n3. Robust\n4. Mặc định (Không chuẩn hóa)")
            scale_type = int(input())
            self.train_scaled, self.scaler = scaler.normalize_data(self.train_filled, scale_type)
            print('#---------Hoàn thành xử lý dữ liệu-----------#')
            return True

    def preprocess(self):
        '''
            Xử lý date
        '''
        self.train_data["Date"] = pd.to_datetime(self.train_data["Date"])
        self.train_data.set_index("Date", inplace=True)
        self.train_data = self.train_data.asfreq("D")

    
import pandas as pd
from processor import Processor
def run():
    print("Nhập đường dẫn data train: ")
    train_path = input()

    try:
        train_df = pd.read_csv(train_path)
        print("Data train có dạng:")
        print(train_df.head())
        print(f"Cỡ: {train_df.shape}\n")
        processor = Processor(train_df)
        done_process = processor.run()
        if not done_process:
            print("#---------------Đóng---------------#")
            return
        

    except Exception as e:
        raise ValueError("Đường dẫn không hợp lệ") from e
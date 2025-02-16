import numpy as np
import matplotlib.pyplot as plt

def replace_and_fill_outliers(train_data, outliers):
        """
        Thay thế outliers trong Series bằng NaN và điền lại bằng các phương pháp khác nhau.

        Parameters:
            outliers (pd.DataFrame): DataFrame chứa outliers (index có thể khác `data.index`).
        """


        valid_outliers_idx = outliers.index.intersection(train_data.index)

        cleaned_data = train_data.copy()

        cleaned_data.loc[valid_outliers_idx] = np.nan
        method = int(input())
        if method == 1:
            filled_data = cleaned_data.fillna(cleaned_data.mean())
        elif method == 2:
            filled_data = cleaned_data.fillna(method="bfill")
        elif method == 3:
            filled_data = cleaned_data.fillna(method="ffill")
        else:
            print("Chọn mặc định fill: interpolate")
            filled_data = cleaned_data.interpolate(method="linear")
        # Vẽ biểu đồ so sánh trước và sau khi xử lý
        plt.figure(figsize=(12, 6))
        plt.plot(train_data, label="Original Data", color='blue', alpha=0.6)
        plt.plot(filled_data, label="Cleaned Data", color='green', linestyle="--")
        plt.scatter(valid_outliers_idx, train_data.loc[valid_outliers_idx], color='red', label="Replaced Outliers", zorder=3)

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Outlier Replacement using {method} method")
        plt.legend()
        plt.show()

        print(f"Thay thế {len(valid_outliers_idx)} outliers và fill: '{method}'.")
        return filled_data
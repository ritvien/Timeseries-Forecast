import pandas as pd
from processor import Processor
from model import Model
import utils
LOG_RESULTS = "../tests/logs/metrics_results.csv"
TEST_PATH = "../tests/test.csv"


def run():
    print("Nhập đường dẫn data train: ")
    train_path = input()

    try:
        train_df = pd.read_csv(train_path)
        print("Data train có dạng:")
        print(train_df.head())
        print(f"Cỡ: {train_df.shape}\n")
    except Exception as e:
        raise ValueError("Đường dẫn không hợp lệ") from e
    

    print("#-----------------------------------DATA PROCESSING------------------------#")
    processor = Processor(train_df)
    done_process = processor.run()
    if not done_process:
        print("#---------------Đóng---------------#")
        return
    
    print("#-----------------------------------MODELING---------------------------#")
    train_preprocessed = processor.train_scaled
    scaler = processor.scaler
    model = Model(train_preprocessed)
    model.run()
    model.fit()


    print("#---------------------------------EVALUATING---------------------------#")
    test_data = pd.read_csv(TEST_PATH)
    test_data["Date"] = pd.to_datetime(test_data["Date"])
    test_data.set_index("Date", inplace=True)
    test_data = test_data.asfreq("D")

    model.predict(len(test_data))

    predictions = model.predictions
    conf_int = model.conf_int

    if scaler is not None:
        y_pred = utils.inverse_scale(predictions, scaler)
        if conf_int is not None:
            conf_int[:,0] = utils.inverse_scale(conf_int[:,0], scaler)
            conf_int[:,1] = utils.inverse_scale(conf_int[:,1], scaler)
    else:
        y_pred = predictions

    

    df_metrics = utils.compute_metrics(model, scaler, test_data.values, y_pred)
    index = utils.save_metrics_to_csv(df_metrics, LOG_RESULTS)
    utils.plot_forecast(processor.train_data, test_data, y_pred, model, index=index, conf_int=conf_int)
    results = pd.read_csv(LOG_RESULTS)
    print(results)
    return



        

        


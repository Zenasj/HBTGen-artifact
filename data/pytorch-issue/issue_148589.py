import torch
from tsfm_public import TimeSeriesForecastingPipeline
from ibm_granite import TinyTimeMixerForPrediction  # Adjust import if necessary

# Assume input_df is a properly formatted DataFrame with a datetime column 'date'
# and an identifier column 'item_id', plus at least one target column 'close'
# For example:
# input_df = pd.read_csv('path_to_your_csv')
# input_df['date'] = pd.to_datetime(input_df['date'])

timestamp_column = "date"
target_columns = ['close']
context_length = 512

zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",  # Using huggingface model
    num_input_channels=len(target_columns),
)

pipeline = TimeSeriesForecastingPipeline(
    zeroshot_model,
    timestamp_column=timestamp_column,  # Column dtype = DateTime
    id_columns=['item_id'],             # Column dtype = String
    target_columns=target_columns,      # Column Type = float
    explode_forecasts=False,
    freq="5min",
    device="mps",  # Setting device to MPS
)

# Trigger inference
zeroshot_forecast = pipeline(input_df)
print(zeroshot_forecast.tail())
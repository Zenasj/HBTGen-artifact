import torch
import pandas as pd

pd.DataFrame({'a': [1]}).to_parquet('crash')
pd.read_parquet('crash')
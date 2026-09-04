import pandas as pd
import time
import os

def log_data(count, risk):
    os.makedirs("data", exist_ok=True)

    df = pd.DataFrame([{
        "time": time.time(),
        "count": count,
        "risk": risk
    }])

    file = "data/logs.csv"

    if not os.path.exists(file):
        df.to_csv(file, index=False)
    else:
        df.to_csv(file, mode='a', header=False, index=False)
from sklearn.preprocessing import StandardScaler
import pandas as pd

def standardize(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return pd.DataFrame(scaler.transform(X), columns = X.columns)

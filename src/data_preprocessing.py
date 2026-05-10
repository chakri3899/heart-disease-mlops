import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    
    # Split features & target
    X = df.drop("num", axis=1)   # UCI target column is 'num'
    y = df["num"]

    # Convert target to binary (0 = no disease, 1 = disease)
    y = (y > 0).astype(int)

    # Impute missing values
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, scaler
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into single dataframe
df = pd.concat([X, y], axis=1)

# Save locally
df.to_csv("data/heart.csv", index=False)

print("Dataset downloaded successfully using UCI repo!")
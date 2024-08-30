import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("bitcoin_prices_final.csv")

df_scaled = df.copy()
df_scaled = df_scaled.drop(columns=["Timestamp"])
scaler = StandardScaler()
df_scaled[df_scaled.columns] = scaler.fit_transform(df_scaled[df_scaled.columns])

price_df = df_scaled

# spotify_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Step 1: Load Data
df = pd.read_csv("spotify.csv")
print("✅ Data Loaded. Shape:", df.shape)

# Step 2: Clean Data
df = df.dropna()
df = df[df['popularity'] > 0]
print("✅ Cleaned. New shape:", df.shape)

# Step 3: Inspect
print(df.head())
print(df.describe())

# Step 4: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[['danceability', 'energy', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Spotify Features")
plt.show()

# Step 5: Scatter Plot — Danceability vs Energy
fig = px.scatter(df, x='danceability', y='energy', color='popularity',
                 hover_data=['track_name', 'artists'])
fig.show()

# Step 6: Select Features & Target
features = ['danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features]
y = df['popularity']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)

# Step 10: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n✅ Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 11: Feature Importance
importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
importance = importance.sort_values(by='Coefficient', ascending=False)
print("\n🎯 Feature Influence on Popularity:\n", importance)

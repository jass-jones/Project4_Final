import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('Resources/merged_dedup.csv')

X = df.drop(columns = ['artist_name', 'track_name', 'track_id', 'top_200', 'key']).dropna()
y = df['top_200'].dropna()

# X_dummy = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))
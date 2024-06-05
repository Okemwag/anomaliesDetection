import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('checkin_data.csv')

# Preprocess the data
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute
data['day'] = data['timestamp'].dt.day

# Create additional features
data['time_since_last_checkin'] = data['timestamp'].diff().dt.total_seconds().fillna(0)
data['same_location_as_previous'] = (data['location_id'] == data['location_id'].shift(1)).astype(int)

# Select features for the model
features = ['hour', 'minute', 'day', 'location_id', 'time_since_last_checkin', 'same_location_as_previous']

X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Predict anomalies
data['anomaly'] = model.predict(scaler.transform(X))

# Anomalies are marked as -1, normal as 1
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})

# Evaluate the results
anomalies = data[data['anomaly'] == 1]
normal = data[data['anomaly'] == 0]

print(f"Number of anomalies detected: {len(anomalies)}")
print(f"Number of normal instances: {len(normal)}")

# Print the anomalies
print("Anomalies detected:")
print(anomalies)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(data['timestamp'], data['location_id'], c=data['anomaly'], cmap='coolwarm')
plt.xlabel('Timestamp')
plt.ylabel('Location ID')
plt.title('Anomaly Detection in Check-In Data')
plt.xticks(rotation=45)
plt.show()

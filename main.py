from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
model = None
scaler = None

def train_model():
    global model, scaler
    data = pd.read_csv('checkin_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    data['day'] = data['timestamp'].dt.day
    data['time_since_last_checkin'] = data['timestamp'].diff().dt.total_seconds().fillna(data['timestamp'].diff().dt.total_seconds().mean())
    data['same_location_as_previous'] = (data['location_id'] == data['location_id'].shift(1)).astype(int)

    features = ['hour', 'minute', 'day', 'location_id', 'time_since_last_checkin', 'same_location_as_previous']
    X = data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_scaled)

@app.route('/train', methods=['POST'])
def train():
    train_model()
    return "Model trained successfully!", 200

@app.route('/predict', methods=['POST'])
def predict():
    checkin_data = request.json
    df = pd.DataFrame(checkin_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day'] = df['timestamp'].dt.day

    if len(df) > 1:
        df['time_since_last_checkin'] = df['timestamp'].diff().dt.total_seconds().fillna(df['timestamp'].diff().dt.total_seconds().mean())
        df['same_location_as_previous'] = (df['location_id'] == df['location_id'].shift(1)).astype(int)
    else:
        df['time_since_last_checkin'] = 0
        df['same_location_as_previous'] = 1

    features = ['hour', 'minute', 'day', 'location_id', 'time_since_last_checkin', 'same_location_as_previous']
    X = df[features]
    X_scaled = scaler.transform(X)
    df['anomaly'] = model.predict(X_scaled)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    anomalies = df[df['anomaly'] == 1]
    response = {
        'anomalies': anomalies.to_dict(orient='records'),
        'total_anomalies': len(anomalies)
    }

    return jsonify(response)

if __name__ == '__main__':
    train_model()  # Train model on startup
    app.run(host='0.0.0.0', port=8080, debug=True)


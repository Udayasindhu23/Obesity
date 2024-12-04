# obesity/obesity_model.py
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_obesity_model():
    # Sample data: Replace this with actual dataset for production.
    data = {
        'height': [5.1, 5.7, 6.0, 5.4, 5.5],
        'weight': [50, 75, 95, 60, 65],
        'obesity': [0, 1, 1, 0, 0]  # 0 = Not Obese, 1 = Obese
    }

    X = np.array([data['height'], data['weight']]).T
    y = np.array(data['obesity'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the model and scaler
    with open('obesity_model.pkl', 'wb') as f:
        pickle.dump((scaler, model), f)

# Train and save the model
train_obesity_model()

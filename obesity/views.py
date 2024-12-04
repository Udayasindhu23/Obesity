# obesity/views.py
from django.shortcuts import render
import pickle
import numpy as np

# Load pre-trained model
with open('obesity_model.pkl', 'rb') as f:
    scaler, model = pickle.load(f)

def home(request):
    if request.method == 'POST':
        height = float(request.POST['height'])
        weight = float(request.POST['weight'])

        # Predict obesity
        input_data = np.array([[height, weight]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        result = 'Obese' if prediction[0] == 1 else 'Not Obese'
        return render(request, 'result.html', {'result': result})
    return render(request, 'home.html')

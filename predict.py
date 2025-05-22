import joblib
from utils import preprocess_text

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input message
msg = input("Enter your SMS message: ")

# Preprocess and predict
processed_msg = preprocess_text(msg)
msg_vec = vectorizer.transform([processed_msg])
prediction = model.predict(msg_vec)

print(f"Prediction: {prediction[0].upper()}")

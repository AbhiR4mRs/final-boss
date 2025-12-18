import joblib

# Load saved model
model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_intent(text):
    text_tfidf = vectorizer.transform([text])
    return model.predict(text_tfidf)[0]

# Test examples
tests = [
    "open spotify",
    "remember I have a meeting tomorrow",
    "what are my tasks",
    "hello friday",
    "stop friday",
]

for t in tests:
    print(f"{t} → {predict_intent(t)}")

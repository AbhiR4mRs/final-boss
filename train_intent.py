import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# 1. Load dataset
# Make sure your CSV has the 200 items for best results
data = pd.read_csv("data/intent_dataset.csv")

X = data["sentence"]
y = data["intent"]

# 2. Train-test split
# Stratify=y is critical to ensure small classes (like 'exit') are in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF Vectorization
# IMPROVEMENT: Removed stop_words. Words like "my", "what", "is" are 
# the primary differentiators for intent classification.
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    strip_accents='unicode'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Model
# IMPROVEMENT: LinearSVC generally outperforms LogisticRegression on small text sets.
# IMPROVEMENT: class_weight='balanced' fixes the "0.00 recall" issue for small classes.
model = LinearSVC(
    class_weight='balanced', 
    random_state=42, 
    max_iter=2000,
    dual=True # Recommended for when n_samples > n_features
)
model.fit(X_train_tfidf, y_train)

# 5. Evaluation
y_pred = model.predict(X_test_tfidf)

print("\n=== Optimized Classification Report ===\n")
# zero_division=0 prevents the warning if a class is still difficult to predict
print(classification_report(y_test, y_pred, zero_division=0))

print("\n=== Confusion Matrix ===\n")
print(confusion_matrix(y_test, y_pred))

# 6. Save model and vectorizer
joblib.dump(model, "models/intent_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")

# 7. Verification Test
def test_input(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

print("\n--- Live Test Verification ---")
print(f"'stop friday' -> {test_input('stop friday')}")
print(f"'what is my deadline' -> {test_input('what is my deadline')}")
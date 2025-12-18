import joblib
from auth.face_auth import FaceAuthenticator
from memory.friday_memory import FridayMemory


# -------------------------------
# Load ML Intent Model
# -------------------------------
intent_model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict_intent(text):
    """Predicts intent using trained ML model"""
    X = vectorizer.transform([text])
    return intent_model.predict(X)[0]


# -------------------------------
# Rule-Based Exit Override (Safety)
# -------------------------------
EXIT_KEYWORDS = ["exit", "quit", "stop", "bye", "goodbye", "shutdown"]


def is_exit_command(text):
    return any(word in text.lower() for word in EXIT_KEYWORDS)


# -------------------------------
# FRIDAY MAIN
# -------------------------------
def main():
    print("\n[FRIDAY] Initializing authentication...\n")

    # Authenticate user
    authenticator = FaceAuthenticator()
    ACCESS_MODE = authenticator.authenticate()

    print(f"[FRIDAY] Access Mode: {ACCESS_MODE}\n")

    # Initialize memory
    memory = FridayMemory()

    print("FRIDAY: Hello! Type your command (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # --- Exit override ---
        if is_exit_command(user_input):
            print("FRIDAY: Goodbye. Have a great day.")
            break

        # --- Predict intent ---
        intent = predict_intent(user_input)

        # -------------------------------
        # Route by intent
        # -------------------------------

        # CHAT (always allowed)
        if intent == "chat":
            print("FRIDAY: I'm here with you. How can I help?")

        # MEMORY STORE (OWNER only)
        elif intent == "memory_store":
            response = memory.store_memory(
                user_input,
                access_mode=ACCESS_MODE
            )
            print(f"FRIDAY: {response}")

        # MEMORY RECALL (OWNER only)
        elif intent == "memory_recall":
            response = memory.recall_memory(
                access_mode=ACCESS_MODE
            )
            print(f"FRIDAY: {response}")

        # SYSTEM COMMAND (BLOCKED FOR NOW)
        elif intent == "system_command":
            if ACCESS_MODE != "OWNER":
                print("FRIDAY: System access is restricted to my owner.")
            else:
                print("FRIDAY: System command recognized (execution disabled in demo).")

        else:
            print("FRIDAY: I'm not sure how to handle that.")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()

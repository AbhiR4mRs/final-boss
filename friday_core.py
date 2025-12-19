import threading
import queue
import time
import joblib

from auth.face_auth import FaceAuthenticator
from memory.friday_memory import FridayMemory
from memory.working_memory import WorkingMemory
from brain_logic import FridayBrain

from speech.stt_whisper import SpeechRecognizer
from voice.friday_voice import FridayVoice


# -------------------------------
# LOAD INTENT MODELS
# -------------------------------
intent_model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

EXIT_KEYWORDS = {"exit", "quit", "bye", "shutdown", "stop"}


def predict_intent(text):
    X = vectorizer.transform([text])
    return intent_model.predict(X)[0]


# -------------------------------
# GLOBAL PIPELINE OBJECTS
# -------------------------------
audio_queue = queue.Queue()
response_queue = queue.Queue()

llm_lock = threading.Lock()
is_speaking = threading.Event()


# -------------------------------
# LISTENER THREAD
# -------------------------------
def listener_loop(stt):
    while True:
        if is_speaking.is_set():
            time.sleep(0.2)
            continue

        text = stt.listen(duration=4)
        if not text:
            continue

        print(f"\nYou: {text}")

        if text.lower() in EXIT_KEYWORDS:
            audio_queue.put("EXIT")
            break

        audio_queue.put(text)


# -------------------------------
# BRAIN THREAD
# -------------------------------
def brain_loop(brain, memory, working_memory, ACCESS_MODE):
    while True:
        user_input = audio_queue.get()

        if user_input == "EXIT":
            response_queue.put("Shutting down. Goodbye.")
            break

        working_memory.add("User", user_input)
        intent = predict_intent(user_input)

        context_memories = ""
        if intent == "memory_recall":
            context_memories = memory.recall_memory(access_mode=ACCESS_MODE)

        if intent == "memory_store":
            result = memory.store_memory(user_input, access_mode=ACCESS_MODE)
            response_queue.put(result)
            continue

        # -------------------------------
        # LLM (SERIALIZED)
        # -------------------------------
        with llm_lock:
            response = brain.generate_response(
                user_input=user_input,
                intent=intent,
                access_mode=ACCESS_MODE,
                memories=context_memories,
                working_context=working_memory.context()
            )

        working_memory.add("FRIDAY", response)
        response_queue.put(response)


# -------------------------------
# SPEAKER THREAD
# -------------------------------
def speaker_loop(voice):
    while True:
        response = response_queue.get()
        is_speaking.set()
        voice.speak(response)
        is_speaking.clear()

        if "shutting down" in response.lower():
            break


# -------------------------------
# MAIN
# -------------------------------
def main():
    authenticator = FaceAuthenticator()
    memory = FridayMemory()
    working_memory = WorkingMemory(max_turns=6)

    brain = FridayBrain(owner_name="Abhiram")
    stt = SpeechRecognizer(model_size="tiny")
    voice = FridayVoice()

    print("[FRIDAY] Authenticating...")
    ACCESS_MODE = authenticator.authenticate()
    print(f"[FRIDAY] Access Mode: {ACCESS_MODE}")

    voice.speak("FRIDAY online.")

    threading.Thread(
        target=listener_loop,
        args=(stt,),
        daemon=True
    ).start()

    threading.Thread(
        target=brain_loop,
        args=(brain, memory, working_memory, ACCESS_MODE),
        daemon=True
    ).start()

    threading.Thread(
        target=speaker_loop,
        args=(voice,),
        daemon=True
    ).start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

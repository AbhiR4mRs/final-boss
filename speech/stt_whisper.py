import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel


class SpeechRecognizer:
    """
    Real-time Speech Recognition using Faster-Whisper
    Optimized for low latency conversational agents
    """

    def __init__(self, model_size="tiny", device="cpu"):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8"
        )
        self.sample_rate = 16000

    def listen(self, duration=4):
        """
        Records audio from microphone and transcribes it
        """
        print("[LISTENING] Speak now...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        audio = np.squeeze(audio)

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True
        )

        text = " ".join(seg.text for seg in segments).strip()
        return text

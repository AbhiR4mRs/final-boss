import sounddevice as sd
from kokoro import KPipeline


class FridayVoice:
    """
    FRIDAY Neural Speech System
    Real-time Kokoro-82M playback (no disk I/O)
    """

    def __init__(self, voice_name="af_sarah", lang="a"):
        self.pipeline = KPipeline(lang_code=lang)
        self.voice_name = voice_name
        self.sample_rate = 24000

    def speak(self, text):
        """
        Streams and plays speech audio in real time
        """
        generator = self.pipeline(
            text,
            voice=self.voice_name,
            speed=1.0
        )

        for _, _, audio in generator:
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()  # ensures smooth playback

from runtime.voice_control import Voice
import time

voice = Voice()
voice.start()

while True:
    cmd = voice.get()
    if cmd:
        print("GOT CMD:", cmd)
    time.sleep(0.1)
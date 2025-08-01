import threading
from playsound import playsound

def play_sound_async(sound_file):
    """
    Plays a sound file in a separate thread to avoid blocking the main thread.
    """
    thread = threading.Thread(target=playsound, args=(sound_file,))
    thread.daemon = True  # Set as a daemon so it exits when the main thread exits
    thread.start()

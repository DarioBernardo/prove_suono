import pyaudio
import wave
import time
import numpy as np
from queue import Queue

from model_builder import load_hotword_recognition_model
import matplotlib.mlab as mlab


chunk_duration = 0.5 # Each read length in seconds from mic.
fs = 44100 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)

def detect_triggerword_spectrum(model, x):
    """
    Function to predict the location of the trigger word.

    Argument:
    x -- spectrum of shape (freqs, Tx)
    i.e. (Number of frequencies, The number time steps)

    Returns:
    predictions -- flattened numpy array to shape (number of output time steps)
    """
    # the spectogram outputs  and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    return predictions.reshape(-1)


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.

    Argument:
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered positive

    Returns:
    True if new trigger word detected in the latest chunk
    """
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False


def get_spectrogram(data):
    """
    Function to compute a spectrogram.

    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
    pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx



class HotWordDetector:

    def __init__(self, ding_filename:str = 'ding.wav'):
        self.model = load_hotword_recognition_model()
        self.filename = ding_filename
        self.p = pyaudio.PyAudio()
        # Queue to communiate between the audio callback and main thread
        self.q = Queue()
        self.run = True
        self.silence_threshold = 150
        # Run the demo for a timeout seconds
        self.timeout = time.time() + 0.5 * 60  # 0.5 minutes from now
        self.data = None

    def play_ding(self):

        # Set chunk size of 1024 samples per data frame
        chunk = 1024

        # Open the sound file
        wf = wave.open(self.filename, 'rb')

        # Create an interface to PortAudio

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        stream = self.p.open(format = self.p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)

        # Read data in chunks
        temp_data = wf.readframes(chunk)

        print("data read")

        # Play the sound by writing the audio data to the stream
        while temp_data != '':
            stream.write(temp_data)
            temp_data = wf.readframes(chunk)
            if temp_data == b'':
                break

        print("played!")

        # Close and terminate the stream
        stream.close()

    def __callback(self, in_data, frame_count, time_info, status):
        if time.time() > self.timeout:
            self.run = False

        data0 = np.frombuffer(in_data, dtype='int16')
        # print(silence_threshold)
        # print(data0.sum())
        # print(np.abs(data0).mean())
        if np.abs(data0).mean() < self.silence_threshold:
            # sys.stdout.write('-')
            print("silence")
            return (in_data, pyaudio.paContinue)
        else:
            # sys.stdout.write('.')
            print("sound detected")
        self.data = np.append(self.data, data0)
        if len(self.data) > feed_samples:
            self.data = self.data[-feed_samples:]
            # Process data async by sending a queue.
            self.q.put(self.data)
        return (in_data, pyaudio.paContinue)

    def listen_for_hotword(self, callback):

        stream = self.get_new_audio_input_stream()
        stream.start_stream()

        try:
            while self.run:
                data = self.q.get()
                spectrum = get_spectrogram(data)
                preds = detect_triggerword_spectrum(self.model, spectrum)
                new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
                if new_trigger:
                    print("activated!")
                    # sys.stdout.write('1')
                    stream.stop_stream()
                    stream.close()
                    self.play_ding()
                    callback()
                    print("restarting")
                    stream = self.get_new_audio_input_stream()
                    stream.start_stream()
                    print("Input stream started")

                # sys.stdout.flush()
        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()
            self.timeout = time.time()
            self.run = False

        stream.stop_stream()
        stream.close()

    def get_new_audio_input_stream(self):
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=fs,
            input=True,
            frames_per_buffer=chunk_samples,
            # input_device_index=0,
            stream_callback=self.__callback)
        return stream

    def __del__(self):
        self.p.terminate()


def do_something():
    print("######### DOING SOMETHING ######")


hot_word_detector = HotWordDetector()
print("Start listening")
hot_word_detector.listen_for_hotword(do_something)
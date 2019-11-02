import pyaudio
import wave

chunk = 1024

p = pyaudio.PyAudio()
wf = wave.open("ding.wav", 'rb')

# open stream based on the wave object which has been input.
stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# read data (based on the chunk size)
data = wf.readframes(chunk)

# play stream (looping from beginning of file to the end)
while data != '':
    # writing to the stream is what *actually* plays the sound.
    stream.write(data)
    data = wf.readframes(chunk)

    if data == b'':
        break
    print('loop')

# cleanup stuff.
stream.close()
p.terminate()
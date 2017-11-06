import tensorflow as tf
import numpy as np
import librosa
import os.path

import Metadata
import subprocess
from soundfile import SoundFile
import Utils

def get_multitrack_placeholders(shape, input_shape=None, name=""):
    '''
    Creates Tensorflow placeholders for mixture, accompaniment, and voice.
    :param shape: Shape of each individual sample
    :return: List of multitrack placeholders for mixture, accompaniment, and voice
    '''
    if input_shape is None:
        input_shape = shape
    m = tf.placeholder(dtype=tf.float32, shape=input_shape, name="mix_input" + name)
    a = tf.placeholder(dtype=tf.float32, shape=shape, name="acc_input" + name)
    v = tf.placeholder(dtype=tf.float32, shape=shape, name="voice_input" + name)
    return m,a,v

def get_multitrack_input(shape, batch_size, name="", input_shape=None):
    '''
    Creates multitrack placeholders and a random shuffle queue based on it
    :param input_shape: Shape of accompaniment and voice magnitudes
    :param batch_size: Number of samples in each batch
    :param name: How to name the placeholders
    :return: [List of mixture,acc,voice placeholders, random shuffle queue, symbolic batch sample from queue]
    '''
    m,a,v = get_multitrack_placeholders(shape, input_shape=input_shape)

    min_after_dequeue = 0
    buffer = 1000
    capacity = min_after_dequeue + buffer

    if input_shape is None:
        input_shape = shape
    queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, [tf.float32, tf.float32, tf.float32], [input_shape, shape, shape])
    input_batch = queue.dequeue_many(batch_size, name="input_batch" + name)

    return [m,a,v], queue, input_batch

def crop(tensor, target_shape):
    '''
    Crops a 4D tensor [batch_size, width, height, channels] along the width and height axes to a target shape.
    :param tensor: 4D tensor [batch_size, width, height, channels] that should be cropped
    :param target_shape: Target shape (4D tensor) that the tensor should be cropped to
    :return: Cropped tensor
    '''
    shape = np.array(tensor.get_shape().as_list())
    diff = shape - np.array(target_shape)
    assert(diff[0] == 0 and diff[3] == 0) # Only two axes can differ
    assert np.sum(diff % 2) == 0 # We have to have the same amount of entries to crop on each side
    assert np.min(diff) >= 0 # Only positive difference allowed
    diff /= 2

    output = tensor
    if diff[1] > 0:
        output = output[:,diff[1]:-diff[1],:,:]
    if diff[2] > 0:
        output = output[:,:,diff[2]:-diff[2],:]
    return output

def load_and_enqueue(sess, model_config, queue, enqueue_op, input_ph, song_list):
    '''
    Executed by an input thread that appends items from the dataset to the input queue
    :param sess: TF session
    :param enqueue_op: Queue enqueue operation
    :param input_ph: Input placeholder
    :param thread_num: Number of thread
    :param max_threads: Total number of threads
    '''

    print("Started input thread")

    input_frames = input_ph[0].get_shape().as_list()[1]
    output_frames = input_ph[1].get_shape().as_list()[1]
    padding = (input_frames - output_frames)/2

    for [mix, acc, voice] in song_list:
        try:
            mix_mag, _ = audioFileToSpectrogram(mix.path, fftWindowSize=model_config["num_fft"], hopSize=model_config["num_hop"], expected_sr=model_config["expected_sr"], buffer=True)
            mix_mag = np.pad(mix_mag, [(0, 0), (padding, padding)], mode='constant', constant_values=0.0) # Pad along time axis

            if isinstance(acc, float):
                acc_mag = np.zeros(mix_mag.shape, np.float32)
            else:
                acc_mag, _ = audioFileToSpectrogram(acc.path, fftWindowSize=model_config["num_fft"], hopSize=model_config["num_hop"], expected_sr=model_config["expected_sr"], buffer=True)
            if isinstance(voice, float):
                voice_mag = np.zeros(mix_mag.shape, np.float32)
            else:
                voice_mag, _ = audioFileToSpectrogram(voice.path, fftWindowSize=model_config["num_fft"], hopSize=model_config["num_hop"], expected_sr=model_config["expected_sr"], buffer=True)
        except Exception as e:
            print("Error while computing spectrogram for file " + mix.path + ". Skipping")
            print(e)
            continue

        # Pad along the frequency axis
        mix_mag = Utils.pad_freqs(mix_mag, input_ph[0].get_shape().as_list()[:2])
        acc_mag = Utils.pad_freqs(acc_mag, input_ph[1].get_shape().as_list()[:2])
        voice_mag = Utils.pad_freqs(voice_mag, input_ph[2].get_shape().as_list()[:2])

        # Partition spectrogram into sections and append to queue, keeping in mind the input might need additional context
        for pos in range(0, acc_mag.shape[1] - output_frames + 1, output_frames):
            sess.run([enqueue_op], feed_dict={input_ph[0] : mix_mag[:,pos:pos+input_frames,np.newaxis],
                                              input_ph[1] : acc_mag[:,pos:pos+output_frames,np.newaxis],
                                              input_ph[2] : voice_mag[:,pos:pos+output_frames,np.newaxis]})
        print("Finished song " + mix.path)

    print("Input thread finished! Closing queue")
    sess.run(queue.close(cancel_pending_enqueues=False))

def random_amplify(magnitude):
    '''
    Randomly amplifies or attenuates the input magnitudes
    :param magnitude: SINGLE Magnitude spectrogram excerpt, or list of spectrogram excerpts that each have their own amplification factor
    :return: Amplified magnitude spectrogram
    '''
    if isinstance(magnitude, np.ndarray):
        return np.random.uniform(0.2, 1.2) * magnitude
    else:
        assert(isinstance(magnitude, list))
        factor = np.random.uniform(0.2, 1.2)
        for i in range(len(magnitude)):
            magnitude[i] = factor * magnitude[i]
        return magnitude

def readWave(audio_path, start_frame, end_frame, mono=True, sample_rate=None, clip=True):
    snd_file = SoundFile(audio_path, mode='r')
    inf = snd_file._info
    audio_sr = inf.samplerate

    snd_file.seek(start_frame)
    audio = snd_file.read(end_frame - start_frame, dtype='float32')
    snd_file.close()
    audio = audio.T # Tuple to numpy, transpose axis to (channels, frames)

    # Convert to mono if desired
    if mono and len(audio.shape) > 1 and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0)

    # Resample if needed
    if sample_rate is not None and sample_rate != audio_sr:
        audio = librosa.resample(audio, audio_sr, sample_rate, res_type="kaiser_fast")
        audio_sr = sample_rate

    # Clip to [-1,1] if desired
    if clip:
        audio = np.minimum(np.maximum(audio, -1.0), 1.0)

    return audio, audio_sr

def readAudio(audio_path, offset=0.0, duration=None, mono=True, sample_rate=None, clip=True, padding_duration=0.0, metadata=None):
    '''
    Reads an audio file wholly or partly, and optionally converts it to mono and changes sampling rate.
    By default, it loads the whole audio file. If the offset is set to None, the duration HAS to be not None,
    and the offset is then randomly determined so that a random section of the audio is selected with the desired duration.
    Optionally, the file can be zero-padded by a certain amount of seconds at the start and end before selecting this random section.

    :param audio_path: Path to audio file
    :param offset: Position in audio file (s) where to start reading. If None, duration has to be not None, and position will be randomly determined.
    :param duration: How many seconds of audio to read
    :param mono: Convert to mono after reading
    :param sample_rate: Convert to given sampling rate if given
    :param padding_duration: Amount of padding (s) on each side that needs to be filled up with silence if it isn't available
    :param metadata: metadata about audio file, accelerates reading audio since duration does not need to be determined from file 
    :return: Audio signal, Audio sample rate
    '''

    if os.path.splitext(audio_path)[1][1:].lower() == "mp3":  # If its an MP3, call ffmpeg with offset and duration parameters
        # Get mp3 metadata information and duration
        if metadata is None:
            audio_sr, audio_channels, audio_duration = Metadata.get_mp3_metadata(audio_path)
        else:
            audio_sr = metadata[0]
            audio_channels = metadata[1]
            audio_duration = metadata[2]
        print(audio_duration)

        pad_front_duration = 0.0
        pad_back_duration = 0.0

        if offset is None:  # In this case, select random section of audio file
            assert (duration is not None)
            max_start_pos = audio_duration+2*padding_duration-duration
            if (max_start_pos <= 0.0):  # If audio file is longer than duration of desired section, take all of it, will be padded later
                print("WARNING: Audio file " + audio_path + " has length " + str(audio_duration) + " but is expected to be at least " + str(duration))
                return librosa.load(audio_path, sample_rate, mono, res_type='kaiser_fast')  # Return whole audio file
            start_pos = np.random.uniform(0.0,max_start_pos) # Otherwise randomly determine audio section, taking padding on both sides into account
            offset = max(start_pos - padding_duration, 0.0) # Read from this position in audio file
            pad_front_duration = max(padding_duration - start_pos, 0.0)
        assert (offset is not None)

        if duration is not None: # Adjust duration if it overlaps with end of track
            pad_back_duration = max(offset + duration - audio_duration, 0.0)
            duration = duration - pad_front_duration - pad_back_duration # Subtract padding from the amount we have to read from file
        else: # None duration: Read from offset to end of file
            duration = audio_duration - offset

        pad_front_frames = int(pad_front_duration * float(audio_sr))
        pad_back_frames = int(pad_back_duration * float(audio_sr))


        args = ['ffmpeg', '-noaccurate_seek',
                '-ss', str(offset),
                '-t', str(duration),
                '-i', audio_path,
                '-f', 's16le', '-']

        audio = []
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=open(os.devnull, 'wb'))
        num_reads = 0
        while True:
            output = process.stdout.read(4096)
            if output == '' and process.poll() is not None:
                break
            if output:
                audio.append(librosa.util.buf_to_float(output, dtype=np.float32))
                num_reads += 1

        audio = np.concatenate(audio)
        if audio_channels > 1:
            audio = audio.reshape((-1, audio_channels)).T

    else: #Not an MP3: Handle with PySoundFile
        # open audio file
        snd_file = SoundFile(audio_path, mode='r')
        inf = snd_file._info
        audio_sr = inf.samplerate

        if duration is not None:
            num_frames = int(duration * float(audio_sr))
        pad_frames = int(padding_duration * float(audio_sr))
        pad_front_frames = 0
        pad_back_frames = 0

        if offset is None:  # In this case, select random section of audio file
            assert (duration is not None)
            max_start_pos = inf.frames + 2 * pad_frames - num_frames
            if (max_start_pos <= 0):  # If audio file is longer than duration of desired section, take all of it, will be padded later
                print("WARNING: Audio file " + audio_path + " has frames  " + str(inf.frames) + " but is expected to be at least " + str(num_frames))
                return librosa.load(audio_path, sample_rate, mono, res_type='kaiser_fast')  # Return whole audio file
            start_pos = np.random.randint(0, max_start_pos)  # Otherwise randomly determine audio section, taking padding on both sides into account
            start_frame = max(start_pos - pad_frames, 0)  # Read from this position in audio file
            pad_front_frames = max(pad_frames - start_pos, 0)
        else:
            start_frame = int(offset * float(audio_sr))

        if duration is not None:  # Adjust duration if it overlaps with end of track
            pad_back_frames = max(start_frame + num_frames - inf.frames, 0)
            num_frames = num_frames - pad_front_frames - pad_back_frames
        else: # Duration is None => Read from start frame to end of track
            num_frames = inf.frames - start_frame

        snd_file.seek(start_frame)
        audio = snd_file.read(num_frames, dtype='float32')
        snd_file.close()
        audio = audio.T  # Tuple to numpy, transpose axis to (channels, frames)

        centre_start_frame = start_frame - pad_front_frames + pad_frames
        centre_end_frame = start_frame + num_frames + pad_back_frames - pad_frames

    # AT THIS POINT WE HAVE A [N_CHANNELS, N_SAMPLES] NUMPY ARRAY FOR THE AUDIO
    # Pad as indicated at beginning and end
    if len(audio.shape) > 1:
        audio = np.pad(audio, [(0,0),(pad_front_frames, pad_back_frames)],mode="constant",constant_values=0.0)
    else:
        audio = np.pad(audio, [(pad_front_frames, pad_back_frames)], mode="constant", constant_values=0.0)

    # Convert to mono if desired
    if mono and len(audio.shape) > 1 and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0)

    # Resample if needed
    if sample_rate is not None and sample_rate != audio_sr:
        audio = librosa.resample(audio, audio_sr, sample_rate, res_type="kaiser_fast")
        audio_sr = sample_rate

    # Clip to [-1,1] if desired
    if clip:
        audio = np.minimum(np.maximum(audio, -1.0), 1.0)

    if float(audio.shape[0])/float(sample_rate) < 1.0:
        print("----------------------ERROR------------------")

    if os.path.splitext(audio_path)[1][1:].lower() == "mp3":
        return audio, audio_sr
    else:
        return audio, audio_sr, centre_start_frame, centre_end_frame

# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioIn, fftWindowSize=1024, hopSize=512, offset=0.0, duration=None, expected_sr=None, buffer=False, padding_duration=0.0, metadata=None):
    '''
    Audio to FFT magnitude and phase conversion. Input can be a filepath to an audio file or a numpy array directly.
    By default, the whole audio is used for conversion. By setting duration to the desired number of seconds to be read from the audio file,
    reading can be sped up.
    For accelerating reading, the buffer option can be activated so that a numpy filedump of the magnitudes
    and phases is created after processing and loaded the next time it is requested.
    :param audioIn: 
    :param fftWindowSize: 
    :param hopSize: 
    :param offset: 
    :param duration: 
    :param expected_sr: 
    :param buffer: 
    :return: 
    '''

    writeNumpy = False
    if isinstance(audioIn, str): # Read from file
        if buffer and os.path.exists(audioIn + ".npy"): # Do we need to load a previous numpy buffer file?
            assert(offset == 0.0 and duration is None) # We can only load the whole buffer file
            with open(audioIn + ".npy", 'r') as file: # Try loading
                try:
                    [magnitude, phase] = np.load(file)
                    return magnitude, phase
                except Exception as e: # In case loading did not work, remember and overwrite file later
                    print("Could not load " + audioIn + ".npy. Loading audio again and recreating npy file!")
                    writeNumpy = True
        audio, sample_rate, _ , _= readAudio(audioIn, duration=duration, offset=offset, sample_rate=expected_sr, padding_duration=padding_duration, metadata=metadata) # If no buffering, read audio file
    else: # Input is already a numpy array
        assert(expected_sr is None and duration is None and offset == 0.0) # Make sure no other options are active
        audio = audioIn

    # Compute magnitude and phase
    spectrogram = librosa.stft(audio, fftWindowSize, hopSize)
    magnitude, phase = librosa.core.magphase(spectrogram)
    phase = np.angle(phase) # from e^(1j * phi) to phi
    assert(np.max(magnitude) < fftWindowSize and np.min(magnitude) >= 0.0)

    # Buffer results if desired
    if (buffer and ((not os.path.exists(audioIn + ".npy")) or  writeNumpy)):
        np.save(audioIn + ".npy", [magnitude, phase])

    return magnitude, phase

def add_audio(audio_list, path_postfix):
    '''
    Reads in a list of audio files, sums their signals, and saves them in new audio file which is named after the first audio file plus a given postfix string
    :param audio_list: List of audio file paths
    :param path_postfix: Name to append to the first given audio file path in audio_list which is then used as save destination
    :return: Audio file path where the sum signal was saved
    '''
    save_path = audio_list[0] + "_" + path_postfix + ".wav"
    if not os.path.exists(save_path):
        for idx, instrument in enumerate(audio_list):
            instrument_audio, sr = librosa.load(instrument, sr=None)
            if idx == 0:
                audio = instrument_audio
            else:
                audio += instrument_audio
        if np.min(audio) < -1.0 or np.max(audio) > 1.0:
            print("WARNING: Mixing tracks together caused the result to have sample values outside of [-1,1]. Clipping those values")
            audio = np.minimum(np.maximum(audio, -1.0), 1.0)

        librosa.output.write_wav(save_path, audio, sr)
    return save_path

def norm(magnitude):
    '''
    Log(1 + magnitude)
    :param magnitude: Input magnitude spectrogram
    :return: Log-normalized magnitude spectrogram
    '''
    return tf.log1p(magnitude)

def denorm(logmagnitude):
    '''
    Exp(logmagnitude) - 1
    :param logmagnitude: Log-normalized magnitude spectrogram
    :return: Unnormalized magnitude spectrogram
    '''
    return tf.expm1(logmagnitude)

def spectrogramToAudioFile(magnitude, fftWindowSize, hopSize, phaseIterations=10, phase=None, length=None):
    '''
    Computes an audio signal from the given magnitude spectrogram, and optionally an initial phase.
    Griffin-Lim is executed to recover/refine the given the phase from the magnitude spectrogram.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param phase: If given, starts ISTFT with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return: 
    '''
    if phase is not None:
        if phaseIterations > 0:
            # Refine audio given initial phase with a number of iterations
            return reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations, phase, length)
        # reconstructing the new complex matrix
        stftMatrix = magnitude * np.exp(phase * 1j) # magnitude * e^(j*phase)
        audio = librosa.istft(stftMatrix, hop_length=hopSize, length=length)
    else:
        audio = reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations)
    return audio

def reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations=10, initPhase=None, length=None):
    '''
    Griffin-Lim algorithm for reconstructing the phase for a given magnitude spectrogram, optionally with a given
    intial phase.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param initPhase: If given, starts reconstruction with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return: 
    '''
    for i in range(phaseIterations):
        if i == 0:
            if initPhase is None:
                reconstruction = np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
            else:
                reconstruction = np.exp(initPhase * 1j) # e^(j*phase), so that angle => phase
        else:
            reconstruction = librosa.stft(audio, fftWindowSize, hopSize)
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == phaseIterations - 1:
            audio = librosa.istft(spectrum, hopSize, length=length)
        else:
            audio = librosa.istft(spectrum, hopSize)
    return audio
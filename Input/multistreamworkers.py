import numpy as np
import Input
from Sample import Sample

class MultistreamWorker_GetSpectrogram:
    @staticmethod
    def run(communication_queue, exit_flag, options):
        '''
        Worker method that reads audio from a given file list and appends the processed spectrograms to the cache queue.
        :param communication_queue: Queue of the cache from which examples are added to the cache
        :param exit_flag: Flag to indicate when to exit the process
        :param options: Audio processing parameters and file list
        '''
        filename_list = options["file_list"]
        num_files = len(filename_list)

        n_fft = options['num_fft']
        hop_length = options['num_hop']

        # Re-seed RNG for this process
        np.random.seed()

        while not exit_flag.is_set():
            # Decide which element to read next randomly
            id_file_to_read = np.random.randint(num_files)
            item = filename_list[id_file_to_read]

            # Calculate the required amounts of padding
            duration_frames = int(options["duration"] * options["expected_sr"])
            padding_duration = options["padding_duration"]

            try:
                if isinstance(item, Sample): # Single audio file: Use metadata to read section from it
                    metadata = [item.sample_rate, item.channels, item.duration]
                    TF_rep, _ = Input.audioFileToSpectrogram(item.path, expected_sr=options["expected_sr"], offset=None, duration=options["duration"], fftWindowSize=n_fft, hopSize=hop_length, padding_duration=options["padding_duration"], metadata=metadata)
                    TF_rep = np.ndarray.astype(TF_rep, np.float32)  # Cast to float32
                    communication_queue.put(Input.random_amplify(TF_rep))

                elif isinstance(item, float): # This means the track is a (not as file existant) silence track so we insert a zero spectrogram
                    TF_rep = np.zeros((n_fft / 2 + 1, duration_frames), dtype=np.float32)
                    TF_rep = np.ndarray.astype(TF_rep, np.float32)  # Cast to float32
                    communication_queue.put(Input.random_amplify(TF_rep))
                else:
                    assert (hasattr(item, '__iter__')) # Supervised case: Item is a list of files to read, starting with the mixture
                    # We want to get the spectrogram of the mixture (first entry in list), and of the sources and store them in cache as one training sample
                    sample = list()

                    file = item[0]
                    metadata = [file.sample_rate, file.channels, file.duration]
                    mix_audio, mix_sr, source_start_frame, source_end_frame = Input.readAudio(file.path, offset=None, duration=options["duration"], sample_rate=options["expected_sr"], padding_duration=padding_duration, metadata=metadata)
                    mix_mag, _ = Input.audioFileToSpectrogram(mix_audio, fftWindowSize=n_fft, hopSize=hop_length)
                    sample.append(mix_mag)

                    for file in item[1:]:
                        if isinstance(file, Sample):
                            #mag, _ = Input.audioFileToSpectrogram(file.path, expected_sr=options["expected_sr"], fftWindowSize=n_fft, hopSize=hop_length, buffer=True)
                            source_audio, _ = Input.readWave(file.path, source_start_frame, source_end_frame, sample_rate=options["expected_sr"])
                            mag, _ = Input.audioFileToSpectrogram(source_audio, fftWindowSize=n_fft, hopSize=hop_length)
                            #mag = mag[:, :options["output_shape"][2]]
                        else:
                            assert(isinstance(file, float)) # This source is silent in this track
                            source_shape = [mix_mag.shape[0], mix_mag.shape[1] - padding*2]
                            mag = np.zeros(source_shape, dtype=np.float32) # Therefore insert zeros
                        sample.append(mag)

                    communication_queue.put(sample)
            except Exception as e:
                print(e)
                print("Error while computing spectrogram for " + item.path + ". Skipping file.")

        # This is necessary so that this process does not block. In particular, if there are elements still in the queue
        # from this process that were not yet 'picked up' somewhere else, join and terminate called on this process will
        # block
        communication_queue.cancel_join_thread()

import pickle
import numpy as np
import tensorflow as tf
import librosa

import Utils
from Input import Input

import Models.WGAN_Critic
import Models.Unet

from mir_eval.separation import validate, bss_eval_sources

def alpha_snr(target, estimate):
    # Compute SNR: 10 log_10 ( ||s_target||^2 / ||s_target - alpha * s_estimate||^2 ), but scale target to get optimal SNR (opt. wrt. alpha)
    # Optimal alpha is Sum_i=1(s_target_i * s_estimate_i) / Sum_i=1 (s_estimate_i ^ 2) = inner_prod / estimate_power
    estimate_power = np.sum(np.square(estimate))
    target_power = np.sum(np.square(target))
    inner_prod = np.inner(estimate, target)
    alpha = inner_prod / estimate_power
    error_power = np.sum(np.square(target - alpha * estimate))
    snr = 10 * np.log10(target_power / error_power)
    return snr

def bss_evaluate(model_config, dataset, load_model):
    '''
    Calculates source separation evaluation metrics of a given separator model on the test set using BSS-Eval
    :param model_config: Separation network configuration required to build symbolic computation graph of network
    :param dataset: Test dataset
    :param load_model: Path to separator model checkpoint containing the network weights
    :return: Dict containing evaluation metrics
    '''
    # Determine input and output shapes, if we use U-net as separator
    freq_bins = model_config["num_fft"] / 2 + 1  # Make even number of freq bins
    disc_input_shape = [1, freq_bins - 1, model_config["num_frames"],1]  # Shape of discriminator input

    separator_class = Models.Unet.Unet(model_config["num_layers"])
    sep_input_shape, sep_output_shape = separator_class.getUnetPadding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Placeholders and input normalisation
    input_ph, queue, [mix_context, acc, voice] = Input.get_multitrack_input(sep_output_shape[1:], 1, name="input_batch", input_shape=sep_input_shape[1:])

    mix = Input.crop(mix_context, sep_output_shape)
    mix_norm, mix_context_norm, acc_norm, voice_norm = Input.norm(mix), Input.norm(mix_context), Input.norm(acc), Input.norm(voice)

    print("Testing...")

    # BUILD MODELS
    # Separator
    separator_acc_norm, separator_voice_norm = separator_func(mix_context_norm, reuse=False)
    separator_acc, separator_voice = Input.denorm(separator_acc_norm), Input.denorm(separator_voice_norm)

    # Start session and queue input threads
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Load model
    # Load pretrained model to continue training, if we are supposed to
    restorer = tf.train.Saver(None, write_version=tf.train.SaverDef.V2)
    print("Num of variables" + str(len(tf.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for testing')

    # Initialize total score object
    song_scores = list()

    for multitrack in dataset:
        filename = multitrack[0].path
        print("Evaluating file " + filename)
        if filename.__contains__("DSD100"):
            db = "DSD100"
        elif filename.__contains__("Kala"):
            db = "IKala"
        elif filename.__contains__("ccmixter"):
            db = "CCMixter"
        elif filename.__contains__("MedleyDB"):
            db = "MedleyDB"
        song_info = {"Title" : filename, "Database" : db}

        # Load mixture and pad it so that output sources have the same length after STFT/ISTFT
        mix_audio, mix_sr = librosa.load(multitrack[0].path, sr=model_config["expected_sr"])
        mix_length = len(mix_audio)
        mix_audio_pad = librosa.util.fix_length(mix_audio, mix_length + model_config["num_fft"] // 2) # Pad input so that ISTFT later leads to same-length audio
        mix_mag, mix_ph = Input.audioFileToSpectrogram(mix_audio_pad, model_config["num_fft"], model_config["num_hop"])
        source_time_frames = mix_mag.shape[1]

        # Preallocate source predictions (same shape as input mixture)
        acc_pred_mag = np.zeros(mix_mag.shape, np.float32)
        voice_pred_mag = np.zeros(mix_mag.shape, np.float32)

        input_time_frames = sep_input_shape[2]
        output_time_frames = sep_output_shape[2]

        # Pad mixture spectrogram across time at beginning and end so that neural network can make prediction at the beginning and end of signal
        pad_time_frames = (input_time_frames - output_time_frames) / 2
        mix_mag = np.pad(mix_mag, [(0,0), (pad_time_frames, pad_time_frames)], mode="constant", constant_values=0.0)

        # Iterate over mixture magnitudes, fetch network prediction
        for source_pos in range(0,source_time_frames,output_time_frames):
            # If this output patch would reach over the end of the source spectrogram, set it so we predict the very end of the output, then stop
            if source_pos + output_time_frames > source_time_frames:
                source_pos = source_time_frames - output_time_frames

            # Prepare mixture excerpt by selecting time interval
            mix_mag_part = mix_mag[:,source_pos:source_pos+input_time_frames]
            mix_mag_part = Utils.pad_freqs(mix_mag_part, sep_input_shape[1:3]) # Pad along frequency axis
            mix_mag_part = mix_mag_part[np.newaxis,:,:,np.newaxis]

            # Fetch network prediction
            acc_mag_part, voice_mag_part = sess.run([separator_acc, separator_voice], feed_dict={mix_context:mix_mag_part})

            # Save predictions
            #source_shape = [1, freq_bins, acc_mag_part.shape[2], 1]
            acc_pred_mag[:,source_pos:source_pos+output_time_frames] = acc_mag_part[0,:-1,:,0]
            voice_pred_mag[:,source_pos:source_pos + output_time_frames] = voice_mag_part[0,:-1,:,0]

        # Spectrograms to audio, using mixture phase
        acc_pred_audio = Input.spectrogramToAudioFile(acc_pred_mag, model_config["num_fft"], model_config["num_hop"], phase=mix_ph, length=mix_length, phaseIterations=0)
        voice_pred_audio = Input.spectrogramToAudioFile(voice_pred_mag, model_config["num_fft"], model_config["num_hop"], phase=mix_ph, length=mix_length, phaseIterations=0)

        # Load original sources
        if isinstance(multitrack[1], float):
            acc_audio = np.zeros(mix_audio.shape, np.float32)
        else:
            acc_audio, _ = librosa.load(multitrack[1].path, sr=model_config["expected_sr"])
        if isinstance(multitrack[2], float):
            voice_audio = np.zeros(mix_audio.shape, np.float32)
        else:
            voice_audio, _ = librosa.load(multitrack[2].path, sr=model_config["expected_sr"])

        # Check if any reference source is completely silent, if so, inject some very slight noise into it to avoid problems during SDR calculation with zero signals
        reference_zero = False
        if np.max(np.abs(acc_audio)) == 0.0:
            acc_audio += np.random.uniform(-1e-10, 1e-10, size=acc_audio.shape)
            reference_zero = True
        if np.max(np.abs(voice_audio)) == 0.0:
            voice_audio += np.random.uniform(-1e-10, 1e-10, size=voice_audio.shape)
            reference_zero = True

        # Evaluate BSS according to MIREX voice separation method # http://www.music-ir.org/mirex/wiki/2016:Singing_Voice_Separation
        ref_sources = np.vstack([acc_audio, voice_audio]) #/ np.linalg.norm(acc_audio + voice_audio) # Normalized audio
        pred_sources = np.vstack([acc_pred_audio, voice_pred_audio]) #/ np.linalg.norm(acc_pred_audio + voice_pred_audio) # Normalized estimates
        validate(ref_sources, pred_sources)
        scores = bss_eval_sources(ref_sources, pred_sources, compute_permutation=False)

        song_info["SDR"] = scores[0]
        song_info["SIR"] = scores[1]
        song_info["SAR"] = scores[2]

        # Compute reference scores and SNR only if both sources are not silent, since they are undefined otherwise
        if not reference_zero:
            mix_ref = np.vstack([mix_audio, mix_audio]) #/ np.linalg.norm(mix_audio + mix_audio)
            mix_scores = bss_eval_sources(ref_sources, mix_ref, compute_permutation=False)
            norm_scores = np.array(scores) - np.array(mix_scores)

            # Compute SNR: 10 log_10 ( ||s_target||^2 / ||s_target - alpha * s_estimate||^2 ), but scale target to get optimal SNR (opt. wrt. alpha)
            voice_snr = alpha_snr(voice_audio, voice_pred_audio)
            acc_snr = alpha_snr(acc_audio, acc_pred_audio)
            voice_ref_snr = alpha_snr(voice_audio, mix_audio)
            acc_ref_snr = alpha_snr(acc_audio, mix_audio)

            song_info["NSDR"] = norm_scores[0]
            song_info["NSIR"] = norm_scores[1]
            song_info["SNR"] = np.array([acc_snr, voice_snr])
            song_info["NSNR"] = np.array([acc_snr - acc_ref_snr, voice_snr - voice_ref_snr])

        song_scores.append(song_info)
        print(song_info)

    with open("evaluation_ " + load_model + "_.pkl", "wb") as file: #TODO proper filename
        pickle.dump(song_scores, file)

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

    return song_scores
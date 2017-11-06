import numpy as np
from lxml import etree
import os.path
import librosa

import Input.Input
from Sample import Sample


def subtract_audio(mix_list, instrument_list):
    '''
    Generates new audio by subtracting the audio signal of an instrument recording from a mixture
    :param mix_list: 
    :param instrument_list: 
    :return: 
    '''

    assert(len(mix_list) == len(instrument_list))
    new_audio_list = list()

    for i in range(0, len(mix_list)):
        new_audio_path = os.path.dirname(mix_list[i]) + os.path.sep + "remainingmix" + os.path.splitext(mix_list[i])[1]
        new_audio_list.append(new_audio_path)

        if os.path.exists(new_audio_path):
            continue
        mix_audio, mix_sr = librosa.load(mix_list[i], mono=False, sr=None)
        inst_audio, inst_sr = librosa.load(instrument_list[i], mono=False, sr=None)
        assert (mix_sr == inst_sr)
        new_audio = mix_audio - inst_audio
        if not (np.min(new_audio) >= -1.0 and np.max(new_audio) <= 1.0):
            print("Warning: Audio for mix " + str(new_audio_path) + " exceeds [-1,1] float range!")

        librosa.output.write_wav(new_audio_path, new_audio, mix_sr) #TODO switch to compressed writing
        print("Wrote accompaniment for song " + mix_list[i])
    return new_audio_list

def create_sample(db_path, instrument_node):
   path = db_path + os.path.sep + instrument_node.xpath("./relativeFilepath")[0].text
   sample_rate = int(instrument_node.xpath("./sampleRate")[0].text)
   channels = int(instrument_node.xpath("./numChannels")[0].text)
   duration = float(instrument_node.xpath("./length")[0].text)
   return Sample(path, sample_rate, channels, duration)

def getDSDFilelist(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    train_vocals, test_vocals, train_mixes, test_mixes, train_accs, test_accs = list(), list(), list(), list(), list(), list()

    for track in tracks:
        # Get mix and vocal instruments
        vocals = create_sample(db_path, track.xpath(".//instrument[instrumentName='Voice']")[0])
        mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
        [acc_path] = subtract_audio([mix.path], [vocals.path])
        acc = Sample(acc_path, vocals.sample_rate, vocals.channels, vocals.duration) # Accompaniment has same signal properties as vocals and mix

        if track.xpath("./databaseSplit")[0].text == "Training":
            train_vocals.append(vocals)
            train_mixes.append(mix)
            train_accs.append(acc)
        else:
            test_vocals.append(vocals)
            test_mixes.append(mix)
            test_accs.append(acc)

    return [train_mixes, train_accs, train_vocals], [test_mixes, test_accs, test_vocals]

def getCCMixter(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    mixes, accs, vocals = list(), list(), list()

    for track in tracks:
        # Get mix and vocal instruments
        voice = create_sample(db_path, track.xpath(".//instrument[instrumentName='Voice']")[0])
        mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
        acc = create_sample(db_path, track.xpath(".//instrument[instrumentName='Instrumental']")[0])

        mixes.append(mix)
        accs.append(acc)
        vocals.append(voice)

    return [mixes, accs, vocals]

def getIKala(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    mixes, accs, vocals = list(), list(), list()

    for track in tracks:
        mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
        orig_path = mix.path
        mix_path = orig_path + "_mix.wav"
        acc_path = orig_path + "_acc.wav"
        voice_path = orig_path + "_voice.wav"

        mix_audio, mix_sr = librosa.load(mix.path, sr=None, mono=False)
        mix.path = mix_path
        librosa.output.write_wav(mix_path, np.sum(mix_audio, axis=0), mix_sr)
        librosa.output.write_wav(acc_path, mix_audio[0,:], mix_sr)
        librosa.output.write_wav(voice_path, mix_audio[1, :], mix_sr)

        voice = create_sample(mix.path, track.xpath(".//instrument[instrumentName='Voice']")[0])
        voice.path = voice_path
        acc = create_sample(mix.path, track.xpath(".//instrument[instrumentName='Instrumental']")[0])
        acc.path = acc_path

        mixes.append(mix)
        accs.append(acc)
        vocals.append(voice)

    return [mixes, accs, vocals]

def getMedleyDB(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text

    mixes, accs, vocals = list(), list(), list()

    tracks = root.xpath(".//track")
    for track in tracks:
        instrument_paths = list()
        # Mix together vocals, if they exist
        vocal_tracks = track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath") + \
                       track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath") + \
                       track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath")
        if len(vocal_tracks) > 0: # If there are vocals, get their file paths and mix them together
            vocal_track = Input.Input.add_audio([db_path + os.path.sep + f.text for f in vocal_tracks], "vocalmix")
            instrument_paths.append(vocal_track)
            vocals.append(Sample.from_path(vocal_track))
        else: # Otherwise append duration of track so silent input can be generated later on-the-fly
            duration = float(track.xpath("./instrumentList/instrument/length")[0].text)
            vocals.append(duration)

        # Mix together accompaniment, if it exists
        acc_tracks = track.xpath(".//instrument[not(instrumentName='Voice') and not(instrumentName='Mix') and not(instrumentName='Instrumental')]/relativeFilepath") #TODO # We assume that there is no distinction between male/female here
        if len(acc_tracks) > 0:  # If there are vocals, get their file paths and mix them together
            acc_track = Input.Input.add_audio([db_path + os.path.sep + f.text for f in acc_tracks], "accmix")
            instrument_paths.append(acc_track)
            accs.append(Sample.from_path(acc_track))
        else:  # Otherwise append duration of track so silent input can be generated later on-the-fly
            duration = float(track.xpath("./instrumentList/instrument/length")[0].text)
            accs.append(duration)

        # Mix together vocals and accompaniment
        mix_track = Input.Input.add_audio(instrument_paths, "totalmix")
        mixes.append(Sample.from_path(mix_track))

    return [mixes, accs, vocals]

def getFMA(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text

    mixes, accs, vocals = list(), list(), list()

    vocal_tracks = root.xpath(".//track/instrumentList/instrument[instrumentName='Mix']")
    instrumental_tracks = root.xpath(".//track/instrumentList/instrument[instrumentName='Instrumental']")
    for instr in vocal_tracks:
        mixes.append(create_sample(db_path,instr))

    for instr in instrumental_tracks:
        mixes.append(create_sample(db_path,instr))
        accs.append(create_sample(db_path,instr))

    return mixes, accs, vocals
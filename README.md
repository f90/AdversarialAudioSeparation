# AdversarialAudioSeparation

Code accompanying the paper "Semi-supervised adversarial audio source separation applied to singing voice extraction" available on arXiv here:

https://arxiv.org/abs/1711.00048

## The idea

Improve existing supervised audio source separation models, which are commonly neural networks, with extra unlabelled mixture recordings as well as unlabelled solo recordings of the sources we want to separate. The network is trained in a normal supervised fashion to minimise its prediction error on fully annotated data (samples with mixture and sources paired up), and at the same time to output source estimates for the extra mixture recordings that are indistinguishable from the solo source recordings.

To achieve this, we use adversarial training: One discriminator network is trained per source to identify whether a source excerpt comes from the real solo source recordings or from the separator when evaluated on the extra mixtures.

This can prevent overfitting to the often small annotated dataset and makes use of the much more easily available unlabelled data.

## Setup

### Requirements

To run the code, the following Python packages are needed. We recommend the GPU version for Tensorflow due to the long running times of this model. You can install them easily using ``pip install -r requirements.txt`` after saving the below list to a text file.

```
tensorflow-gpu>=1.2.0  
sacred>=0.7.0  
audioread>=2.1.5
imageio>=2.2.0
librosa>=0.5.1
lxml>=3.8.0
mir_eval>=0.4
scikits.audiolab>=0.11.0
soundfile>=0.9.0
```

Furthermore, ffmpeg needs to be installed and in your path in case you want to read in mp3 files directly.

### Dataset preparation

Before the code is runnable, the datasets need to be prepared and integrated into the data loading process.
The simpler way to do this is to use the same datasets as used in the experiment in the paper, the alternative to use your own datasets and split them into custom partitions. Please see below and the Training.py code comments for guidance.

When the code is run for the first time, it creates a dataset.pkl file containing the dataset structure after reading in the dataset, so that subsequent starts are much faster.

#### Option 1: Recreate experiment from the paper

If you want to recreate the experiment from the paper, download the datasets DSD100, MedleyDB, CCMixter, and iKala separately.
Then edit the corresponding XML files provided in this repository (DSD100.xml etc.), so that the XML entry

``
<databaseFolderPath>/mnt/daten/Datasets/DSD100</databaseFolderPath>
``

contains the location of the root folder of the respective dataset. Save the file changes and then execute Training.py.

#### Option 2: Use your own data of choice

To use your own datasets and dataset partitioning into supervised, unsupervised, validation and test sets, you can replace the data loading code in Training.py with a custom dataset loading function.

The only requirement to this function is its output format. The output should be a dictionary that maps the following strings to the respective dataset partitions:

```
"train_sup" : sample_list
"train_unsup" : [mix_list, source1_list, source2_list]
"train_valid" : sample_list
"train_test" : sample_list
```

A sample_list is a list with each element being a tuple containing three Sample objects. The order for these objects is mixture, source 1, source 2.
You can initialise Sample objects with the constructor of the Sample class found in ``Sample.py``. Each represents an audio signal along with its metadata. This audio should be preferably in .wav format for fast on-the-fly reading, but other formats such as mp3 are also supported.

The entry for `"train_unsup"` is different since recordings are not paired - instead, this entry is a list containing three lists. These contain mixtures, source1 and source2 Sample objects respectively. The lists can be of different length. since they are not paired.

### Configuration and hyperparameters

You can configure settings and hyperparameters by modifying the ``model_config`` dictionary defined in the beginning of ``Training.py`` or using the commandline features of sacred by setting certain values when calling the script via commandline (see Sacred documentation).

Note that alpha and beta (hyperparameters from the paper) as loss weighting parameters are relatively important for good performance, tweaking these might be necessary. These are also editable in the ``model_config`` dictionary.

## Training

The code is run by executing

``
python Training.py
``

It will train the same separator network first in a purely supervised way, and then using our semi-supervised adversarial approach. Each time, validation performance is measured regularly and early stopping is used, before the final test set performance is evaluated. For the semi-supervised approach, the additional data from ``dataset["train_unsup"]`` is used to improve performance.

Finally, BSS evaluation metrics are computed on the test dataset (SDR, SIR, SAR) - this saves the results in a pickled file along with the name of the dataset, so if you aim to use different datasets, the function needs to be extended slightly. 

Logs are written continuously to the logs subfolder, so training can be supervised with Tensorboard. Checkpoint files of the model are created whenever validation performance is tested.

from sacred import Experiment
import tensorflow as tf
import threading
import numpy as np
import os

import Datasets
from Input import Input as Input
from Input import batchgenerators as batchgen
import Models.WGAN_Critic
import Models.Unet
import Utils
import cPickle as pickle
import Test

ex = Experiment('Adversarial_Source_Separation')

@ex.config
def cfg():
    model_config = {"model_base_dir" : "checkpoints", # Base folder for model checkpoints
                    "log_dir" : "logs", # Base folder for logs files
                    "batch_size" : 64, # Batch size
                    "alpha" : 0.01, # Weighting for adversarial loss (unsupervised)
                    "beta" : 0.01, # Weighting for additive penalty (unsupervised)
                    "lam" : 10, # Weighting term lambda for WGAN gradient penalty
                    "init_disc_lr" : 5e-5, # Discriminator(s) learning rate
                    "init_sup_sep_lr" : 5e-5, # Supervised separator learning rate
                    "init_unsup_sep_lr" : 5e-5, # Unsupervised separator learning rate
                    "epoch_it" : 50, # Number of supervised separator steps per epoch
                    "num_disc": 5,  # Number of discriminator iterations per separator update
                    "num_frames" : 64, # DESIRED number of time frames in the spectrogram per sample (this can be increased when using U-net due to its limited output sizes)
                    "num_fft" : 512, # FFT Size
                    "num_hop" : 256, # FFT Hop size
                    'expected_sr' : 8192, # Downsample all audio input to this sampling rate
                    'mono_downmix' : True, # Whether to downsample the audio input
                    'cache_size' : 64, # Number of audio excerpts that are cached to build batches from
                    'num_workers' : 4, # Number of processes reading audio and filling up the cache
                    "duration" : 10, # Duration in seconds of the audio excerpts in the cache (excluding input context)
                    'min_replacement_rate' : 8,  # roughly: how many cache entries to replace at least per batch on average. Can be fractional
                    'num_layers' : 4, # How many U-Net layers
                    }

    experiment_id = np.random.randint(0,1000000)

@ex.capture
def test(model_config, audio_list, model_folder, load_model):
    # Determine input and output shapes, if we use U-net as separator
    freq_bins = model_config["num_fft"] / 2 + 1  # Make even number of freq bins
    disc_input_shape = [model_config["batch_size"], freq_bins-1, model_config["num_frames"],1]  # Shape of discriminator input
    separator_class = Models.Unet.Unet(model_config["num_layers"])
    sep_input_shape, sep_output_shape = separator_class.getUnetPadding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Placeholders and input normalisation
    input_ph, queue, [mix_context, acc, voice] = Input.get_multitrack_input(sep_output_shape[1:], model_config["batch_size"], name="input_batch", input_shape=sep_input_shape[1:])
    enqueue_op = queue.enqueue(input_ph)

    mix = Input.crop(mix_context, sep_output_shape)
    mix_norm, mix_context_norm, acc_norm, voice_norm = Input.norm(mix), Input.norm(mix_context), Input.norm(acc), Input.norm(voice)

    print("Testing...")

    # BUILD MODELS
    # Separator
    separator_acc_norm, separator_voice_norm = separator_func(mix_context_norm, reuse=False)

    # Supervised objective
    sup_separator_loss = tf.reduce_mean(tf.square(separator_voice_norm - voice_norm)) + tf.reduce_mean(tf.square(separator_acc_norm - acc_norm))

    tf.summary.scalar("sup_sep_loss", sup_separator_loss, collections=['sup', 'unsup'])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)

    # Start session and queue input threads
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep +  model_folder, graph=sess.graph)

    thread = threading.Thread(target=Input.load_and_enqueue, args=(sess, model_config, queue, enqueue_op, input_ph, audio_list))
    thread.deamon = True
    thread.start()

    # CHECKPOINTING
    # Load pretrained model to test
    restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    print("Num of variables" + str(len(tf.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for testing')

    # Start training loop
    _global_step = sess.run(global_step)
    print("Starting!")
    batches = 0
    total_loss = 0.0
    run = True
    while run:
        try:
            _sup_separator_loss = sess.run(
               sup_separator_loss)
            total_loss += _sup_separator_loss # Aggregate loss measure
            batches += 1
        except Exception as e:
            print("Emptied queue - finished this epoch!")
            run = False

    mean_mse_loss = total_loss / float(batches)
    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=mean_mse_loss)])
    writer.add_summary(summary, global_step=_global_step)

    writer.flush()
    writer.close()

    print("Finished testing - Mean MSE: " + str(mean_mse_loss))

    thread.join()

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

    return mean_mse_loss


@ex.capture
def train(model_config, sup_dataset, model_folder, unsup_dataset=None, load_model=None):
    # Determine input and output shapes
    freq_bins = model_config["num_fft"] / 2 + 1  # Make even number of freq bins
    disc_input_shape = [model_config["batch_size"], freq_bins - 1, model_config["num_frames"],1]  # Shape of discriminator input

    separator_class = Models.Unet.Unet(model_config["num_layers"])
    sep_input_shape, sep_output_shape = separator_class.getUnetPadding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Placeholders and input normalisation
    mix_context,acc,voice = Input.get_multitrack_placeholders(sep_output_shape, sep_input_shape, "sup")
    mix = Input.crop(mix_context, sep_output_shape)
    mix_norm, mix_context_norm, acc_norm, voice_norm = Input.norm(mix), Input.norm(mix_context), Input.norm(acc), Input.norm(voice)

    if unsup_dataset != None:
        mix_context_u,acc_u,voice_u = Input.get_multitrack_placeholders(sep_output_shape, sep_input_shape, "unsup")
        mix_u = Input.crop(mix_context_u, sep_output_shape)
        mix_norm_u, mix_context_norm_u, acc_norm_u, voice_norm_u = Input.norm(mix_u), Input.norm(mix_context_u), Input.norm(acc_u), Input.norm(voice_u)

    # Creating the batch generators
    padding_durations = [float(sep_input_shape[2] - sep_output_shape[2]) * model_config["num_hop"] / model_config["expected_sr"] / 2.0, 0, 0]  # Input context that the input audio has to be padded with while reading audio files
    sup_batch_gen = batchgen.BatchGen_Paired(
        model_config,
        sup_dataset,
        sep_input_shape,
        sep_output_shape,
        padding_durations[0]
    )

    # Creating unsupervised batch generator if needed
    if unsup_dataset != None:
        unsup_batch_gens = list()
        for i in range(3):
            shape = (sep_input_shape if i==0 else sep_output_shape)
            unsup_batch_gens.append(batchgen.BatchGen_Single(
                model_config,
                unsup_dataset[i],
                shape,
                padding_durations[i]
            ))

    print("Training...")

    # BUILD MODELS
    # Separator
    separator_acc_norm, separator_voice_norm = separator_func(mix_context_norm, reuse=False)
    separator_acc, separator_voice = Input.denorm(separator_acc_norm), Input.denorm(separator_voice_norm)
    if unsup_dataset != None:
        separator_acc_norm_u, separator_voice_norm_u = separator_func(mix_context_norm_u, reuse=True)
        separator_acc_u, separator_voice_u = Input.denorm(separator_acc_norm_u), Input.denorm(separator_voice_norm_u)
        mask_loss_u = tf.reduce_mean(tf.square(mix_u - separator_acc_u - separator_voice_u))
    mask_loss = tf.reduce_mean(tf.square(mix - separator_acc - separator_voice))

    # SUMMARIES FOR INPUT AND SEPARATOR
    tf.summary.scalar("mask_loss", mask_loss, collections=["sup", "unsup"])
    if unsup_dataset != None:
        tf.summary.scalar("mask_loss_u", mask_loss_u, collections=["unsup"])
        tf.summary.scalar("acc_norm_mean_u", tf.reduce_mean(acc_norm_u), collections=["acc_disc"])
        tf.summary.scalar("voice_norm_mean_u", tf.reduce_mean(voice_norm_u), collections=["voice_disc"])
        tf.summary.scalar("acc_sep_norm_mean_u", tf.reduce_mean(separator_acc_norm_u), collections=["acc_disc"])
        tf.summary.scalar("voice_sep_norm_mean_u", tf.reduce_mean(separator_voice_norm_u), collections=["voice_disc"])
    tf.summary.scalar("acc_norm_mean", tf.reduce_mean(acc_norm), collections=['sup'])
    tf.summary.scalar("voice_norm_mean", tf.reduce_mean(voice_norm), collections=['sup'])
    tf.summary.scalar("acc_sep_norm_mean", tf.reduce_mean(separator_acc_norm), collections=['sup'])
    tf.summary.scalar("voice_sep_norm_mean", tf.reduce_mean(separator_voice_norm), collections=['sup'])

    tf.summary.image("sep_acc_norm", separator_acc_norm, collections=["sup", "unsup"])
    tf.summary.image("sep_voice_norm", separator_voice_norm, collections=["sup", "unsup"])

    # BUILD DISCRIMINATORS, if unsupervised training
    unsup_separator_loss = 0
    if unsup_dataset != None:
        disc_func = Models.WGAN_Critic.dcgan

        # Define real and fake inputs for both discriminators - if separator output and dsicriminator input shapes do not fit perfectly, we will do a centre crop and only discriminate that part
        acc_real_input = Input.crop(acc_norm_u, disc_input_shape)
        acc_fake_input = Input.crop(separator_acc_norm_u, disc_input_shape)
        voice_real_input = Input.crop(voice_norm_u, disc_input_shape)
        voice_fake_input = Input.crop(separator_voice_norm_u, disc_input_shape)

        #WGAN
        acc_disc_loss, acc_disc_real, acc_disc_fake, acc_grad_pen, acc_wasserstein_dist = \
            Models.WGAN_Critic.create_critic(model_config, real_input=acc_real_input, fake_input=acc_fake_input, scope="acc_disc", network_func=disc_func)
        voice_disc_loss, voice_disc_real, voice_disc_fake, voice_grad_pen, voice_wasserstein_dist = \
            Models.WGAN_Critic.create_critic(model_config, real_input=voice_real_input, fake_input=voice_fake_input, scope="voice_disc", network_func=disc_func)

        L_u = - tf.reduce_mean(voice_disc_fake)  - tf.reduce_mean(acc_disc_fake) # WGAN based loss for separator (L_u in paper)
        unsup_separator_loss = model_config["alpha"] * L_u + model_config["beta"] * mask_loss_u # Unsupervised loss for separator: WGAN-based loss L_u and additive penalty term (mask loss), weighted by alpha and beta (hyperparameters)

    # Supervised objective: MSE in log-normalized magnitude space
    sup_separator_loss = tf.reduce_mean(tf.square(separator_voice_norm - voice_norm)) + \
                         tf.reduce_mean(tf.square(separator_acc_norm - acc_norm))

    separator_loss = sup_separator_loss + unsup_separator_loss # Total separator loss: Supervised + unsupervised loss

    # TRAINING CONTROL VARIABLES
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
    disc_lr = tf.get_variable('disc_lr', [],
                              initializer=tf.constant_initializer(model_config["init_disc_lr"], dtype=tf.float32), trainable=False)
    unsup_sep_lr = tf.get_variable('unsup_sep_lr', [],
                             initializer=tf.constant_initializer(model_config["init_sup_sep_lr"], dtype=tf.float32), trainable=False)
    sup_sep_lr = tf.get_variable('sup_sep_lr', [],
                             initializer=tf.constant_initializer(model_config["init_sup_sep_lr"], dtype=tf.float32),
                             trainable=False)

    # Set up optimizers
    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))

    acc_disc_vars, voice_disc_vars = Utils.getTrainableVariables("acc_disc"), Utils.getTrainableVariables("voice_disc")
    print("Voice_Disc_Vars: " + str(Utils.getNumParams(voice_disc_vars)))
    print("Acc_Disc_Vars: " + str(Utils.getNumParams(acc_disc_vars)))

    if unsup_dataset != None:
        with tf.variable_scope("voice_disc_solver"):
            voice_disc_solver = tf.train.AdamOptimizer(learning_rate=disc_lr).minimize(voice_disc_loss, var_list=voice_disc_vars, colocate_gradients_with_ops=True)
        with tf.variable_scope("acc_disc_solver"):
            acc_disc_solver = tf.train.AdamOptimizer(learning_rate=disc_lr).minimize(acc_disc_loss, var_list=acc_disc_vars, colocate_gradients_with_ops=True)
        with tf.variable_scope("unsup_separator_solver"):
            unsup_separator_solver = tf.train.AdamOptimizer(learning_rate=unsup_sep_lr).minimize(
                separator_loss, var_list=separator_vars, colocate_gradients_with_ops=True)
    else:
        with tf.variable_scope("separator_solver"):
            sup_separator_solver = (tf.train.AdamOptimizer(learning_rate=sup_sep_lr).minimize(sup_separator_loss, var_list=separator_vars, colocate_gradients_with_ops=True))

    # SUMMARIES FOR DISCRIMINATORS AND LOSSES
    acc_disc_summaries = tf.summary.merge_all(key="acc_disc")
    voice_disc_summaries = tf.summary.merge_all(key="voice_disc")
    tf.summary.scalar("sup_sep_loss", sup_separator_loss, collections=['sup', "unsup"])
    tf.summary.scalar("unsup_sep_loss", unsup_separator_loss, collections=['unsup'])
    tf.summary.scalar("sep_loss", separator_loss, collections=["sup", "unsup"])
    sup_summaries = tf.summary.merge_all(key='sup')
    unsup_summaries = tf.summary.merge_all(key='unsup')

    # Start session and queue input threads
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep + model_folder, graph=sess.graph)

    print("Starting worker")
    sup_batch_gen.start_workers()
    print("Started worker!")

    if unsup_dataset != None:
        for gen in unsup_batch_gens:
            print("Starting worker")
            gen.start_workers()
            print("Started worker!")

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        print("Num of variables" + str(len(tf.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)


    # Start training loop
    run = True
    _global_step = sess.run(global_step)
    _init_step = _global_step
    it = 0
    while run:
        if unsup_dataset != None:
            # TRAIN DISCRIMINATORS
            for disc_it in range(model_config["num_disc"]):
                    batches = list()
                    for gen in unsup_batch_gens:
                        batches.append(gen.get_batch())

                    _,  _acc_disc_summaries = sess.run(
                        [acc_disc_solver, acc_disc_summaries],
                        feed_dict={mix_context_u: batches[0], acc_u: batches[1]}
                    )

                    _,  _voice_disc_summaries = sess.run(
                        [voice_disc_solver, voice_disc_summaries],
                        feed_dict={mix_context_u: batches[0], voice_u: batches[2]}
                    )

                    writer.add_summary(_acc_disc_summaries, global_step=it)
                    writer.add_summary(_voice_disc_summaries, global_step=it)

                    it += 1

        # TRAIN SEPARATOR
        sup_batch = sup_batch_gen.get_batch()

        if unsup_dataset != None:
            # SUP + UNSUPERVISED TRAINING
            unsup_batches = list()
            for gen in unsup_batch_gens:
                unsup_batches.append(gen.get_batch())

            _, _unsup_summaries, _sup_summaries = sess.run(
                [unsup_separator_solver, unsup_summaries, sup_summaries],
                feed_dict={mix_context: sup_batch[0], acc: sup_batch[1], voice: sup_batch[2],
                           mix_context_u: unsup_batches[0], acc_u:unsup_batches[1], voice_u:unsup_batches[2]}
            )
            writer.add_summary(_unsup_summaries, global_step=_global_step)
        else:
            # PURELY SUPERVISED TRAINING
            _, _sup_summaries = sess.run(
               [sup_separator_solver, sup_summaries],
                feed_dict={mix_context: sup_batch[0], acc: sup_batch[1], voice: sup_batch[2]})
            writer.add_summary(_sup_summaries, global_step=_global_step)

        # Increment step counter, check if maximum iterations per epoch is achieved and stop in that case
        _global_step = sess.run(global_step.assign(_global_step + 1))

        if _global_step - _init_step > model_config["epoch_it"]:
            run = False
            print("Finished training phase, stopping batch generators")
            sup_batch_gen.stop_workers()

            if unsup_dataset != None:
                for gen in unsup_batch_gens:
                    gen.stop_workers()

    # Epoch finished - Save model
    print("Finished epoch!")
    save_path = saver.save(sess, model_config["model_base_dir"] + os.path.sep + model_folder + os.path.sep + model_folder, global_step=int(_global_step))

    # Close session, clear computational graph
    writer.flush()
    writer.close()
    sess.close()
    tf.reset_default_graph()

    return save_path

@ex.capture
def optimise(experiment_id, dataset, supervised):
    '''
    Performs either supervised or unsupervised training of the separation system.
    Training stops if validation loss did not improve for a number of epochs, then final performance on test set is determined 
    :param dataset: Dataset dict containing the supervised, unsupervised, valiation and test partition
    :param supervised: Boolean, whether to train supervised or semi-supervised
    :return: [path to checkpoint file of best model, test loss of best model]
    '''
    if supervised:
        unsup_dataset = None
        model_folder = str(experiment_id) + "_sup"
    else:
        model_folder = str(experiment_id) + "_semisup"
        unsup_dataset = dataset["train_unsup"]

    epoch = 0
    best_loss = 10000
    model_path = None
    worse_epochs = 0
    best_model_path = ""
    while worse_epochs < 1: # Early stopping on validation set after a few epochs
        print("EPOCH: " + str(epoch))
        model_path = train(sup_dataset=dataset["train_sup"], unsup_dataset=unsup_dataset, model_folder=model_folder, load_model=model_path)
        curr_loss = test(audio_list=dataset["valid"], model_folder=model_folder, load_model=model_path)
        epoch += 1
        if curr_loss < best_loss:
            worse_epochs = 0
            print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss))
            best_model_path = model_path
            best_loss = curr_loss
        else:
            worse_epochs += 1
            print("Performance on validation set worsened to " + str(curr_loss))
    print("TRAINING FINISHED - TESTING WITH BEST MODEL " + best_model_path)
    test_loss = test(audio_list=dataset["test"], model_folder=model_folder, load_model=best_model_path)
    return best_model_path, test_loss

@ex.automain
def dsd_100_experiment(model_config):
    # Set up data input
    if os.path.exists('dataset.pkl'):
        with open('dataset.pkl', 'r') as file:
            dataset = pickle.load(file)
        print("Loaded dataset from pickle!")
    else:
        '''
        Create dataset structure comprised of supervised, unsupervised, validation and test partitions
        
        # MODIFY BELOW TO INSERT DSD100, MedleyDB, CCMixter datasets.
        # Each returned item from the dataset reading function (dsd_train, dsd_test, mdb, ccm, ikala) has to be of the following structure:
        # List of 3 elements, each of which is a
        #       List of Sample objects (instantiations of the Sample class - you need to create these as part of your dataset reading function). Each sample represents an audio track
        # The list of 3 elements has to be in order: A list of mixtures, accompaniments and voices, in that order.
        # Example: dsd_train[1] - List of Sample objects, each sample object represents an accompaniment audio file
        # The lists have to be matched: The mixture audio at dsd_train[0][20] has to have its accompaniment at dsd_train[1][20] and voice at dsd_train[2][20]
        # This means that for iKala and MedleyDB you need to generate separate vocal and accompaniment audio manually first!
        # For MedleyDB, we mix together stems with/without vocals to generate the vocal and accompaniment track respectively, then add those signals together for the mixture track, to ensure mix=acc+voice
_       '''

        ###################### MODIFY BELOW

        dsd_train, dsd_test = Datasets.getDSDFilelist("/mnt/daten/PycharmProjects/SingingDBAnnotations/DSD100.xml")
        mdb = Datasets.getMedleyDB("/mnt/daten/PycharmProjects/SingingDBAnnotations/MedleyDB.xml")
        ccm = Datasets.getCCMixter("/mnt/daten/PycharmProjects/SingingDBAnnotations/CCMixter.xml")
        ikala = Datasets.getIKala("/mnt/daten/PycharmProjects/SingingDBAnnotations/iKala.xml")

        ###################### MODIFY ABOVE

        # Draw randomly from datasets
        dataset = dict()
        dataset["train_sup"] = dsd_train # 50 training tracks from DSD100 as supervised dataset
        dataset["train_unsup"] = [list(), list(), list()] # Initialise unsupervised dateaset structure (fill up later)
        dataset["valid"] = [dsd_test[0][:25], dsd_test[1][:25], dsd_test[2][:25]] # Validation and test contains 25 songs of DSD each, plus more (added later)
        dataset["test"] = [dsd_test[0][25:], dsd_test[1][25:], dsd_test[2][25:]]

        # Go through MedleyDB, CCMixter, iKala
        for ds in [mdb, ccm, ikala]:
            num = len(ds[0]) // 3
            # Split dataset in three equal-sized parts, and assign each to the unsupervised dataset, validation and test dataset respectively
            for i in range(3):
                # Only add more examples to unsupervised and test sets, since in this experiment we care about preventing overfitting to the supervised dataset
                dataset["train_unsup"][i].extend(ds[i][:num])
                dataset["valid"][i].extend(ds[i][num:2 * num])
                dataset["test"][i].extend(ds[i][2 * num:])

        ### DELETE ALL OF THE ABOVE IF YOU WANT TO  USE YOUR OWN DATASETS OR PARTITIONING,
        ### AND READ IN YOUR OWN DATASET OBJECTS ACCORDING TO THE RULES SHOWN ABOVE, THEN ASSIGN THEM TO THE DATASET DICT WITH
        ### ENTRIES train_sup, valid and test, RESPECTIVELY.

        # Zip up all paired dataset partitions so we have (mixture, accompaniment, voice) tuples
        dataset["train_sup"] = zip(dataset["train_sup"][0], dataset["train_sup"][1], dataset["train_sup"][2])
        dataset["valid"] = zip(dataset["valid"][0], dataset["valid"][1], dataset["valid"][2])
        dataset["test"] = zip(dataset["test"][0], dataset["test"][1], dataset["test"][2])

        with open('dataset.pkl', 'wb') as file:
            pickle.dump(dataset, file)
        print("Created dataset structure")

    # Optimize in a +supervised fashion until validation loss worsens
    #sup_model_path = "checkpoints/382441_sup/382441_sup-306"
    sup_model_path, sup_loss = optimise(dataset=dataset, supervised=True)
    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))
    sup_scores = Test.bss_evaluate(model_config, dataset=dataset["test"],load_model=sup_model_path)
    print(sup_scores)

    # Train same network architecture semi-supervised
    unsup_model_path, unsup_loss = optimise(dataset=dataset, supervised=False)
    print("Unsupervised training finished! Performance: " + str(unsup_loss))
    unsup_scores = Test.bss_evaluate(model_config, dataset=dataset["test"],load_model=unsup_model_path)
    print(unsup_scores)
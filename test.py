from model import unet
import numpy as np
import tensorflow as tf  # Tensorflow 1.3 (CPU or GPU)
import scipy.io as sio
import os
import time
import configparser

# Saves MAT file with:  (1) R2s prediction (2) S0 Prediction
def generate_results(saved_weights_file, MAT, output_path, patient, GPU):
    echos = np.abs(np.expand_dims(sio.loadmat(MAT)["echos"], axis=0))  # (1,y,x,x,echos)
    mask = np.expand_dims(sio.loadmat(MAT)["brainmsk"], axis=0)  # (1,y,x,z)
    echos = np.multiply(echos, mask[:, :, :, :, None])  # Masks each echo

    # Change Dimensions around so that slices is axis 0
    mask = np.swapaxes(mask, 0, 3)  # (y,x,z,1)
    echos = np.squeeze(np.swapaxes(echos, 0, 3), axis=3)  # (y,x,z,echos)

    # Normalizing Code ######
    normalizing_echo = echos[:, :, :, 0:1]
    normalizing_echo = normalizing_echo * mask
    normalize_echo_mean = np.mean(normalizing_echo[normalizing_echo != 0])
    echos = echos / normalize_echo_mean  # Normalize
    ##########################

    if not os.path.exists(output_path):
        print("Creating Output Path")
        os.makedirs(output_path)
        print(" ( *** CREATED *** )")

    with tf.Session() as sess:
        # Restores the model weights
        new_saver = tf.train.Saver()
        new_saver.restore(sess, (saved_weights_file))
        print("DONE")
        print("Trained Weights Restored")

        # ****** OPTION 1 NO GPU: Loop Through Each Slice *******
        if not GPU:
            R2s = np.zeros((echos.shape[0], echos.shape[1], echos.shape[2]))
            S0 = np.zeros((echos.shape[0], echos.shape[1], echos.shape[2]))

            start = time.time()
            for z in range(echos.shape[0]):
                print(z)
                R2s_z, S0_z = sess.run((UNET.R2s_Pred, UNET.S0_Pred),
                                       feed_dict={UNET.noisy_inputs: echos[z:z + 1], UNET.mask: mask[z:z + 1],
                                                  UNET.dropout_level: 0.0})
                R2s[z] = np.squeeze(R2s_z, axis=3)
                S0[z] = np.squeeze(S0_z, axis=3)
            end = time.time()
            print(str(end - start) + " SECONDS")
        else: # ******* OPTION 2 GPU DO ALL SLICES AT ONCE ********
            R2s, S0 = sess.run((UNET.R2s_Pred, UNET.S0_Pred),feed_dict={UNET.input_echos: echos, UNET.mask: mask, UNET.dropout_level: 0.0})
        # ******************************************************************

        # R2s and S0 predictions have shape (z, y, x), reshape to normal format (y, x, z)
        R2s = np.squeeze(np.swapaxes(np.expand_dims(R2s, axis=3), 0, 3), axis=0)
        S0 = np.squeeze(np.swapaxes(np.expand_dims(S0, axis=3), 0, 3), axis=0) * normalize_echo_mean

        print(np.mean(R2s))

        sio.savemat(output_path + patient + ".mat", {"S0_Pred": S0, "R2s_Pred": R2s})
        print("Done")

######Read in network parameters
config = configparser.ConfigParser()
config.read('config.ini')

x_dim = int(config["networkparams"]["x_dim"])
y_dim = int(config["networkparams"]["y_dim"])
echos = int(config["networkparams"]["num_echos"])

slice_with_echos_shape = (None, y_dim, x_dim, echos)
slice_shape = (None, y_dim, x_dim, 1)
up_down_times = int(config["networkparams"]["up_down_times"])
conv_times = int(config["networkparams"]["conv_times"])
filters_root = int(config["networkparams"]["filters_root"])

os.environ["CUDA_VISIBLE_DEVICES"] = config["global"]["gpu_index"] #GPU we want to use

# Creates instance of empty network without weights loaded
tf.reset_default_graph()
UNET = unet(up_down_times, conv_times, filters_root, slice_with_echos_shape, slice_shape)

#Parameters
#(1) Path to saved model
#(2) Path to the data (echos and mask) you wish to generate R2* for
#(3) Path to save the result (4) Filename of result
#(5) Use GPU to run network
generate_results(r"/path/to/model/myModel.ckpt", "/path/to/data/1.mat",
                      "/path/to/save/", "Patient1", GPU = True)
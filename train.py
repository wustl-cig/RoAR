import tensorflow as tf
import numpy as np
from model import unet
import os
import scipy.io as sio
import configparser

#Saves validation slices to MAT files
def save_to_MAT(sess, network, X, clean_X, mask, F, path, epoch, val_tracking):
    for val_index in range(val_tracking.shape[0]):
        val = val_tracking[val_index]
        X_P = np.expand_dims(X[val], axis=0)
        F_P = np.expand_dims(F[val], axis=0)
        X_clean_P = np.expand_dims(clean_X[val], axis=0)
        mask_P = np.expand_dims(mask[val], axis=0)

        # Queries the info
        S0_Pred, R2s_Pred = sess.run((network.S0_Pred, network.R2s_Pred),
                              feed_dict={network.input_echos: X_P, network.optimize_echos: X_clean_P, network.mask: mask_P,
                                         network.F: F_P, network.learning_rate: 0.0, network.dropout_level: 0.0})

        #Remove Channels and Batch Dimensions from S0 and R2*
        S0_Pred = np.squeeze(S0_Pred, axis=3)
        S0_Pred = np.squeeze(S0_Pred, axis=0)
        R2s_Pred = np.squeeze(R2s_Pred, axis=3)
        R2s_Pred = np.squeeze(R2s_Pred, axis=0)

        e4 = np.squeeze(X_P[:,:,:,4], axis=0) #Fourth echo
        e4_clean = np.squeeze(X_clean_P[:,:,:,4], axis=0) #Fourth echo

        final_path = path + "/epoch_" + str(epoch) + "/"
        if not os.path.exists(final_path):
            print("We are making this path")
            os.makedirs(final_path)
        final_path = final_path + str(val_index)

        sio.savemat(final_path,
        {"Image_" + str(val_index) + "_S0_Prediction": S0_Pred,
         "Image_" + str(val_index) + "_R2s_Prediction": R2s_Pred,
         "Image_" + str(val_index) + "_Echo4" : e4,
         "Image_" + str(val_index) + "_Clean_Echo4_": e4_clean
         })

#Displays and Returns L2 loss of the network
def display_metrics_batched(sess,X,X_clean,mask,F, network, batch_size = 50):
    avg_l2 = 0.0

    for index in range(0,X.shape[0], batch_size):  # Goes through every point 300 at a time
        if index + batch_size <= X.shape[0]:
            batch_X = X[index:index + batch_size]
            batch_clean_X = X_clean[index:index + batch_size]
            batch_mask = mask[index:index + batch_size]
            batch_F = F[index:index + batch_size]
        else:
            batch_X = X[index:]
            batch_clean_X = X_clean[index:]
            batch_mask = mask[index:]
            batch_F = F[index:]

        l2 = sess.run((network.loss), feed_dict={network.input_echos: batch_X, network.optimize_echos: batch_clean_X, network.mask: batch_mask, network.F: batch_F,
                                                                                     network.learning_rate: 0.0, network.dropout_level: 0.0})
        avg_l2 += l2 * batch_X.shape[0]

    avg_l2 /= X.shape[0]

    #Display Metric Values
    print("L2 -->" + str(avg_l2))
    print("------------------------------------------------")
    return avg_l2

#Shuffles the training data
def shuffle_train_data(train_X, train_clean_X, train_mask, train_F):
    permutation = np.random.permutation(train_X.shape[0]) #New indices
    train_X = train_X[permutation]
    train_clean_X = train_clean_X[permutation]
    train_mask = train_mask[permutation]
    train_F =  train_F[permutation]

    return train_X,train_clean_X, train_mask, train_F

#Trains the network
def train_net(train_X, train_clean_X, train_mask, train_F, val_X, val_clean_X, val_mask, val_F,
              learning_rate, epochs, batch_size, drop, lr_schedule, config):

    #Saving Parameters
    save_model_epoch = int(config["save_params"]["save_model_epoch"])
    save_mat_epoch = int(config["save_params"]["save_mat_epoch"])

    #Validation Slices Too Track
    val_tracking = np.array([int(x) for x in config["save_params"]["save_val_slices"].split(",")])

    #Summaries used to record scalar metrics
    l2_train_summ = tf.Summary()
    l2_train_summ.value.add(tag = 'Training L2', simple_value=None)
    # -----------------------
    l2_val_summ = tf.Summary()
    l2_val_summ.value.add(tag = 'Val L2', simple_value=None)

    x_dim = int(config["networkparams"]["x_dim"])
    y_dim = int(config["networkparams"]["y_dim"])
    echos = int(config["networkparams"]["num_echos"])

    slice_with_echos_shape = (None, y_dim, x_dim, echos)
    slice_shape = (None, y_dim, x_dim, 1)
    up_down_times = int(config["networkparams"]["up_down_times"])
    conv_times = int(config["networkparams"]["conv_times"])
    filters_root = int(config["networkparams"]["filters_root"])

    UNET = unet(up_down_times, conv_times, filters_root, slice_with_echos_shape, slice_shape)

    #Runs the graph and starts training
    with tf.Session() as sess:

        saver = tf.train.Saver() #used to save the model
        writer = tf.summary.FileWriter(log_path, sess.graph) #records info for tensorboard
        sess.run(tf.global_variables_initializer())

        #Initial Metrics
        print("Initial Training")
        l2_train = display_metrics_batched(sess,train_X, train_clean_X,train_mask, train_F, UNET)
        print("Initial Validation")
        l2_val = display_metrics_batched(sess,val_X,val_clean_X, val_mask, val_F, UNET)

        # Saves Initial Metrics to Tensorboard
        l2_train_summ.value[0].simple_value = l2_train
        l2_val_summ.value[0].simple_value = l2_val

        writer.add_summary(l2_train_summ, 0)
        writer.add_summary(l2_val_summ, 0)

        for epoch in range(0, epochs+1): #Main loop training for "epochs" epochs
            print(epoch)

            #Decrease learning rate based on epoch
            if epoch in lr_schedule:
                learning_rate = learning_rate / lr_schedule[epoch]

            train_X, train_clean_X, train_mask, train_F = shuffle_train_data(train_X, train_clean_X, train_mask, train_F) # Shuffles Training Data

            for index in range(0,train_X.shape[0],batch_size): #Main Training Loop

                echo_input = train_X[index:index+batch_size] #(batch, y, x, echos)
                echo_optimize = train_clean_X[index:index+batch_size]
                mask_point = train_mask[index:index+batch_size]
                F_point = train_F[index:index+batch_size]

                sess.run((UNET.train_operation),
                         feed_dict={UNET.input_echos: echo_input, UNET.optimize_echos: echo_optimize, UNET.mask: mask_point, UNET.F: F_point,
                                    UNET.learning_rate:learning_rate, UNET.dropout_level: drop}) #hyperparameters

            # Initial Metrics
            print("Training")
            l2_train = display_metrics_batched(sess, train_X, train_clean_X, train_mask, train_F, UNET)
            print("Validation")
            l2_val = display_metrics_batched(sess, val_X, val_clean_X, val_mask, val_F, UNET)

            # Saves Initial Metrics to Tensorboard
            l2_train_summ.value[0].simple_value = l2_train
            l2_val_summ.value[0].simple_value = l2_val

            writer.add_summary(l2_train_summ, epoch)
            writer.add_summary(l2_val_summ, epoch)

            if epoch % save_model_epoch == 0: #Saves the model
                print("Saving Validation Tracking to MAT")
                save_to_MAT(sess, UNET, val_X, val_clean_X, val_mask, val_F, MAT_path, epoch,val_tracking)

            if epoch % save_mat_epoch == 0: #Saves mat results of validation data
                save_model_epoch_path = save_model_path + str(epoch) + "/"
                os.makedirs(save_model_epoch_path)
                save_path = saver.save(sess,save_model_epoch_path + "myModel.ckpt")
                print("Model saved in path: %s" % save_path)

        writer.close()


#Returns original gradient echos and corresponding echos with added noise, F functions, and masks
def prepare_noisy_data(inds, config):

    #Load In Config Info
    input_clean_x_list = []
    input_mask_list = []
    input_F_list = []
    input_noise_x_list = []

    x_dim = int(config["networkparams"]["x_dim"])
    y_dim = int(config["networkparams"]["y_dim"])
    echos = int(config["networkparams"]["num_echos"])

    data_path = config["data"]["data_path"]

    insert_at_ind = 0
    for patient in sorted(inds): #Goes through each patient
        standard_echo = int(config["data"]["standardize_denoise"])

        if standard_echo == 0:
            S0_pt = np.expand_dims(np.abs(sio.loadmat(data_path + str(patient))['S0']), axis=0) # (1,y,x,z)
            S0_pt = np.swapaxes(S0_pt, 0,3) # (1,y,x,z)

        mask_pt = np.expand_dims(np.abs(sio.loadmat(data_path + str(patient) + ".mat")['brainmsk']), axis=0)
        F_pt = np.expand_dims(np.abs(sio.loadmat(data_path + str(patient) + ".mat")['Ffun']),axis=0) #(1,y,x,z,echos)
        X_pt = np.expand_dims(np.abs(sio.loadmat(data_path + str(patient) + ".mat")['echos']),axis=0) #(1,y,x,z,echos)

        mask_pt = np.swapaxes(mask_pt, 0, 3)  # (z, y, x, 1)
        F_pt = np.squeeze(np.swapaxes(F_pt, 0, 3), axis=3)
        X_pt = np.squeeze(np.swapaxes(X_pt, 0, 3), axis=3)

        # Masks F so that all values masked out in echos are 1 in F
        F_pt = np.multiply(F_pt, mask_pt)
        one_mask = np.copy(mask_pt).astype('float32')
        one_mask[one_mask == 1] = -1
        one_mask[one_mask == 0] = 1
        one_mask[one_mask == -1] = 0
        F_pt = np.add(F_pt, one_mask)
        one_mask = None
        F_pt[F_pt > 1.3] = 1.0  # Cap values that may be too high

        #Find the standardizing value for the added noise
        if standard_echo == 0:
            S0_pt = np.multiply(S0_pt, mask_pt)
            Noise_Standardizer = np.mean(S0_pt[S0_pt != 0])
        else:
            standard_echo_array = X_pt[:,:,:,standard_echo]
            Noise_Standardizer = np.mean(standard_echo_array[standard_echo_array != 0])

        z_dimension_for_patient = X_pt.shape[0]

        X_pt = np.multiply(X_pt, mask_pt)
        four_noise_X = np.concatenate((X_pt,X_pt,X_pt,X_pt), axis=0)
        four_clean_x = np.copy(four_noise_X)
        four_mask = np.concatenate((mask_pt,mask_pt,mask_pt,mask_pt), axis=0)
        four_F = np.concatenate((F_pt,F_pt,F_pt,F_pt), axis=0)

        # For each copy add a random level of noise within the indicated range
        for i in range(4):
            low = i * z_dimension_for_patient
            high = (i * z_dimension_for_patient) + z_dimension_for_patient
            SNR = np.random.uniform(float(config["data"]["lower_bound"]), float(config["data"]["upper_bound"]))
            noise = np.random.normal(0, Noise_Standardizer / SNR, X_pt.shape)
            noisyCopy = four_noise_X[low: high] + noise

            noisyS1 = noisyCopy[:,:,:,0:1]
            noisyS1 = noisyS1 * mask_pt
            noisyS1Mean = np.mean(noisyS1[noisyS1 != 0])

            noisyCopy = noisyCopy / noisyS1Mean

            noisyCopy = noisyCopy * mask_pt
            four_clean_x[low: high] = four_clean_x[low:high] / noisyS1Mean
            four_noise_X[low : high] = noisyCopy

        input_clean_x_list.append(four_clean_x)
        input_mask_list.append(four_mask)
        input_F_list.append(four_F)
        input_noise_x_list.append(four_noise_X)

        # Converts Giant List Into NumPy
        numSlices = 0
        for i in range(len(input_clean_x_list)):
            print(i)
            numSlices += input_clean_x_list[i].shape[0]

        input_clean_X = np.zeros((numSlices, y_dim, x_dim, echos))  # Empty Arrays To Be Filled
        input_noise_X = np.zeros((numSlices, y_dim, x_dim, echos))  # Empty Arrays To Be Filled
        input_mask = np.zeros((numSlices, y_dim, x_dim, 1))
        input_F = np.zeros((numSlices, y_dim, x_dim, echos))

        insert_at_ind = 0
        for i in range(len(input_clean_x_list)):
            numSlicesGenerated = input_clean_x_list[i].shape[0]
            input_clean_X[insert_at_ind:insert_at_ind + numSlicesGenerated] = input_clean_x_list[i]
            input_noise_X[insert_at_ind:insert_at_ind + numSlicesGenerated] = input_noise_x_list[i]
            input_mask[insert_at_ind:insert_at_ind + numSlicesGenerated] = input_mask_list[i]
            input_F[insert_at_ind:insert_at_ind + numSlicesGenerated] = input_F_list[i]
            insert_at_ind += numSlicesGenerated

    input_noise_x_list = None
    input_mask_list = None
    input_F_list = None
    input_clean_x_list = None

    return input_noise_X, input_clean_X, input_mask, input_F

# Returns echos and corresponding F-functions and Masks
def grab_data(inds,config):

    input_x_list = []
    input_mask_list = []
    input_F_list = []

    x_dim = int(config["networkparams"]["x_dim"])
    y_dim = int(config["networkparams"]["y_dim"])
    echos = int(config["networkparams"]["num_echos"])

    data_path = config["data"]["data_path"]

    insert_at_ind = 0
    for patient in sorted(inds): #Goes through each patient
        print("Patient: " + str(patient))

        mask_pt = np.expand_dims(np.abs(sio.loadmat(data_path + str(patient) + ".mat")['brainmsk']), axis = 0)
        F_pt = np.expand_dims(np.abs(sio.loadmat(data_path + str(patient) + ".mat")['Ffun']), axis = 0)
        X_pt = np.expand_dims(np.abs(sio.loadmat(data_path + str(patient) + ".mat")['echos']), axis = 0)

        mask_pt = np.swapaxes(mask_pt, 0,3) #(z, y, x, 1)
        F_pt = np.squeeze(np.swapaxes(F_pt, 0,3) ,axis=3) #(z,y,x,1,echos)
        X_pt = np.squeeze(np.swapaxes(X_pt, 0,3) ,axis=3)

        #Mask F
        F_pt = np.multiply(F_pt, mask_pt) #set outer values to 0
        one_mask = np.copy(mask_pt).astype('float32')
        one_mask[one_mask == 1] = -1
        one_mask[one_mask == 0] = 1
        one_mask[one_mask == -1] = 0
        F_pt = np.add(F_pt, one_mask)
        one_mask = None
        F_pt[F_pt > 1.3] = 1.0 #Cap Wrong Values

        z_dimension_for_patient = X_pt.shape[0]
        numSlicesGenerated = X_pt.shape[0]  # Number of slices generated from patient

        X_pt = np.multiply(X_pt, mask_pt)
        normalzing_echo = X_pt[:, :, :, 0:1]
        normalzing_echo = normalzing_echo * mask_pt
        normalize_echo_mean = np.mean(normalzing_echo[normalzing_echo != 0])
        X_pt = X_pt / normalize_echo_mean #Normalize

        input_x_list.append(X_pt)
        input_mask_list.append(mask_pt)
        input_F_list.append(F_pt)

        insert_at_ind += numSlicesGenerated

    #Converts Giant List Into NumPy
    numSlices = 0
    for i in range(len(input_x_list)):
        print(i)
        numSlices += input_x_list[i].shape[0]

    input_X = np.zeros((numSlices, y_dim, x_dim, echos))  # Empty Arrays To Be Filled
    input_mask = np.zeros((numSlices, y_dim, x_dim, 1))  # Does not
    input_F = np.zeros((numSlices, y_dim, x_dim, echos))

    insert_at_ind = 0
    for i in range(len(input_x_list)):
        numSlicesGenerated = input_x_list[i].shape[0]  # Number of slices generated from patient

        input_X[insert_at_ind:insert_at_ind + numSlicesGenerated] = input_x_list[i]
        input_mask[insert_at_ind:insert_at_ind + numSlicesGenerated] = input_mask_list[i]
        input_F[insert_at_ind:insert_at_ind + numSlicesGenerated] = input_F_list[i]
        insert_at_ind += numSlicesGenerated

    input_x_list = None
    input_mask_list = None
    input_F_list = None

    return input_X, input_X, input_mask, input_F


# Loads in echos, masks and F-functions
def load_data(train_pts, val_pts, config):

    #If denoising add noise to data used in optimization
    if bool(config["data"]["denoise"]):
        val_X, val_clean_X, val_mask, val_F = prepare_noisy_data(val_pts, config)
    else:
        val_X, val_clean_X, val_mask, val_F = grab_data(val_pts, config)

    val_mask = np.squeeze(val_mask, axis=3)  # (#pts, y, x)
    val_clean_X = np.multiply(val_clean_X, val_mask[:, :, :, None])
    val_mask = np.expand_dims(val_mask, axis=3)  # (tp, y, x, 1)

    if bool(config["data"]["denoise"]):
        train_X, train_clean_X, train_mask, train_F = prepare_noisy_data(train_pts,config)
    else:
        train_X, train_clean_X, train_mask, train_F = grab_data(train_pts, config)

    train_mask = np.squeeze(train_mask, axis=3)  # (tp, y, x)
    train_clean_X = np.multiply(train_clean_X, train_mask[:, :, :, None])
    train_mask = np.expand_dims(train_mask, axis=3)  # (tp, y, x, 1)

    #Throw away completely masked slices (near top and bottom of MRI)
    keep_train_inds = []
    for ind in range(train_mask.shape[0]):
        maxM = np.amax(train_mask[ind])
        if maxM == 0:
            print("Will Remove: " + str(ind))
        else:
            keep_train_inds.append(ind)

    keep_train_inds = np.array(keep_train_inds)

    train_X = train_X[keep_train_inds]
    train_clean_X = train_clean_X[keep_train_inds]
    train_mask = train_mask[keep_train_inds]
    train_F = train_F[keep_train_inds]

    keep_val_inds = []
    for ind in range(val_mask.shape[0]):
        maxM = np.amax(val_mask[ind])
        if maxM == 0:
            print("Will Remove: " + str(ind))
        else:
            keep_val_inds.append(ind)

    keep_val_inds = np.array(keep_val_inds)

    val_X = val_X[keep_val_inds]
    val_clean_X = val_clean_X[keep_val_inds]
    val_mask = val_mask[keep_val_inds]
    val_F = val_F[keep_val_inds]


    return train_X, train_clean_X, train_mask, train_F, val_X, val_clean_X, val_mask, val_F

#######################################################
#######################################################
#######################################################
##### Runner

config = configparser.ConfigParser()
config.read('config.ini')

#Data Parameters
train_pts = [int(x) for x in config['data']['train_pts'].split(',')]
val_pts = [int(x) for x in config['data']['val_pts'].split(',')]
data_path = config['data']['data_path']

#Load Hyperparameters
lr = float(config["hyperparams"]["learning_rate"])
epochs = int(config["hyperparams"]["epochs"])
batch_size_train = int(config["hyperparams"]["batch_size_train"])
drop = float(config["hyperparams"]["dropout"])

#GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = config["global"]["gpu_index"] #GPU we want to use

tf.reset_default_graph()

#Loads Data In
train_X, train_clean_X, train_mask, train_F, val_X, val_clean_X, val_mask, val_F = load_data(train_pts, val_pts, config)

#Create paths for saving model, logs and validation points
# Create All Paths To Be Used
path = config["save_params"]["save_path"] + config["global"]["experiment_name"] + "/"  # Makes little fodlder for everything we want :)
save_model_path = path + "saved_models" + "/"
log_path = path + "logs/"
MAT_path = path + "MAT/"
paths = [path, save_model_path, log_path, MAT_path]

for path in paths:
    print("---")
    print(path)
    if not os.path.exists(path):
        print("We are making this path")
        os.makedirs(path)
        print(" ( *** CREATED *** )")

#Learning rate decrease schedule
lr_schedule =   {6:1.5,
               20:3.0,
               35:2.0,
               55:2.0,
               150:2.0}

train_net(train_X, train_clean_X, train_mask, train_F, val_X, val_clean_X, val_mask,
          val_F, lr, epochs, batch_size_train, drop, lr_schedule, config)

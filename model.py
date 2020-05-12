import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import ReLU, Dropout, Input, Concatenate, MaxPool2D, Conv2D, Conv2DTranspose

kernel_size = 3 #Each Neurons Spatial Field In Regards To the Previous Activation Maps

def conv3d_relu_dropout(input_, filters_, kernel_size_, dropout_level):
    output_ = Conv2D(filters=filters_, kernel_size=kernel_size_, padding='same')(input_)
    output_ = ReLU()(output_)
    output_ = Dropout(rate=dropout_level)(output_)
    return output_

def conv3d_transpose_relu_dropout(input_, filters_, kernel_size_, dropoout_level):
    output_ = Conv2DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=(2, 2))(input_)
    output_ = ReLU()(output_)
    output_ = Dropout(rate=dropoout_level)(output_)
    return output_

class unet:
    def __init__(self, up_down_times, conv_times, filters_root, slice_with_echos_shape, slice_shape):

        #Inputs to the graph
        self.te = tf.convert_to_tensor(np.array(range(4,44,4)) / 1e3, dtype="float32") #Converts to tensor so it can be used

        self.optimize_echos = tf.placeholder(tf.float32, slice_with_echos_shape, name="Network_Optimize")
        self.input_echos = tf.placeholder(tf.float32, slice_with_echos_shape, name="Network_Input")
        self.max = tf.placeholder(tf.float32, (None, 1,1,1))

        self.mask = tf.placeholder(tf.float32, slice_shape, name="Mask")
        self.F = tf.placeholder(tf.float32, slice_with_echos_shape, name = "F")

        #Hyperparameter inputs to the graph
        self.dropout_level = tf.placeholder(tf.float32, (), name="Dropout_Level") #Level of Dropout Wanted. 1 if testing
        self.learning_rate = tf.placeholder(tf.float32, (), name="Learning_Rate")

        self.skip_layers_storage = [] #stores the layers that will be appended

        # Build Network
        self.net = conv3d_relu_dropout(self.input_echos, filters_root, kernel_size, self.dropout_level)

        for layer in range(up_down_times): #Smushed information down with pooling
            filters = 2 ** layer * filters_root
            for i in range(0, conv_times):
                self.net = conv3d_relu_dropout(self.net, filters, kernel_size, self.dropout_level)
            self.skip_layers_storage.append(self.net)
            self.net = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(self.net)

        filters = 2 ** up_down_times * filters_root
        for i in range(0, conv_times):
            self.net = conv3d_relu_dropout(self.net, filters, kernel_size, self.dropout_level)

        for layer in range(up_down_times - 1, -1, -1): #Transpose Convolves Information To Larger Tensors
            filters = 2 ** layer * filters_root
            self.net = conv3d_transpose_relu_dropout(self.net, filters, kernel_size, self.dropout_level)
            self.net = Concatenate(axis=-1)([self.net, self.skip_layers_storage[layer]])
            for i in range(0, conv_times):
                self.net = conv3d_relu_dropout(self.net, filters, kernel_size, self.dropout_level)

        #Network Outputs
        self.All_Predictions = Conv2D(filters=2 , kernel_size=kernel_size, padding='same', name = "ALL")(self.net) #(1,y, x, 2)
        self.S0_Pred = tf.multiply(self.All_Predictions[:,:,:,0,None], self.mask) ### CHANGE BACK
        self.R2s_Pred = tf.multiply(self.All_Predictions[:,:,:,1,None], self.mask)

        #Metrics to be reported NOT TRAINED ON
        self.physical_equation = tf.multiply(tf.multiply(tf.math.exp(tf.multiply(tf.math.negative(self.R2s_Pred), self.te)), self.S0_Pred), self.F)

        self.loss_matrix = tf.math.square(tf.math.subtract(self.physical_equation, self.optimize_echos))

        self.loss = tf.math.reduce_mean(self.loss_matrix)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_operation = self.optimizer.minimize(loss = self.loss)


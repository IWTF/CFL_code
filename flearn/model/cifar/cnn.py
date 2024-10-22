import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential


def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(3072,)))
    # Reshape Layer
    model.add(Reshape((32, 32, 3)))

    # Conv1 Layer
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(32, 3, padding='same', use_bias=True, activation='relu'))
    model.add(MaxPool2D((2, 2)))
    # Conv2 Layer
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPool2D((2, 2)))
    # Conv3 Layer
    model.add(Conv2D(64, 3, activation='relu'))

    # Flatten Layer
    model.add(Flatten())
    # Dense Layer
    model.add(Dense(64, 'relu'))
    # Output Layer
    model.add(
        Dense(100, 'softmax',  kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model


def construct_model(trainer_type, lr=0.003):
    if trainer_type == 'fedavg':
        return _construct_client_model(lr)
    else:
        return _construct_client_model(lr)

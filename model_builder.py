from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

# Use 1101 for 2sec input audio
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

# Use 272 for 2sec input audio
Ty = 1375# The number of time steps in the output of our model


def get_model_structure(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization
    X = Activation('relu')(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)

    return model


def load_hotword_recognition_model():
    model = get_model_structure(input_shape = (Tx, n_freq))
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    model = load_model('./model/tr_model.h5')

    return model
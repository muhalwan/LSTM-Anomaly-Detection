from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import HeNormal
import numpy as np


def build_lstm_autoencoder_base(input_shape):
    model = Sequential()
    seed = np.random.randint(0, 10000)

    # Encoder
    model.add(Bidirectional(LSTM(64,
                                 return_sequences=True,
                                 kernel_initializer=HeNormal(seed=seed),
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 input_shape=input_shape)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Bottleneck
    model.add(LSTM(32,
                   return_sequences=False,
                   kernel_initializer=HeNormal(seed=seed + 1),
                   kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(RepeatVector(input_shape[0]))

    # Decoder
    model.add(Bidirectional(LSTM(32,
                                 return_sequences=True,
                                 kernel_initializer=HeNormal(seed=seed + 2),
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64,
                                 return_sequences=True,
                                 kernel_initializer=HeNormal(seed=seed + 3),
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(input_shape[1], kernel_initializer=HeNormal(seed=seed + 4))))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

    return model

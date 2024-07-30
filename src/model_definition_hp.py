from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, RepeatVector, \
    TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import HeNormal
import numpy as np


def build_lstm_autoencoder_hp(input_shape, hp):
    model = Sequential()
    seed = np.random.randint(0, 10000)

    # Encoder
    model.add(Bidirectional(LSTM(hp.Int('units_1', 32, 128, step=32),
                                 return_sequences=True,
                                 kernel_initializer=HeNormal(seed=seed),
                                 kernel_regularizer=l1_l2(l1=hp.Float('l1_1', 1e-6, 1e-3, sampling='log'),
                                                          l2=hp.Float('l2_1', 1e-6, 1e-3, sampling='log')),
                                 input_shape=input_shape)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)))

    # Bottleneck
    model.add(LSTM(hp.Int('units_2', 16, 64, step=16),
                   return_sequences=False,
                   kernel_initializer=HeNormal(seed=seed + 1),
                   kernel_regularizer=l1_l2(l1=hp.Float('l1_2', 1e-6, 1e-3, sampling='log'),
                                            l2=hp.Float('l2_2', 1e-6, 1e-3, sampling='log'))))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', 0, 0.5, step=0.1)))
    model.add(RepeatVector(input_shape[0]))

    # Decoder
    model.add(Bidirectional(LSTM(hp.Int('units_2', 16, 64, step=16),
                                 return_sequences=True,
                                 kernel_initializer=HeNormal(seed=seed + 2),
                                 kernel_regularizer=l1_l2(l1=hp.Float('l1_2', 1e-6, 1e-3, sampling='log'),
                                                          l2=hp.Float('l2_2', 1e-6, 1e-3, sampling='log')))))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', 0, 0.5, step=0.1)))

    model.add(Bidirectional(LSTM(hp.Int('units_1', 32, 128, step=32),
                                 return_sequences=True,
                                 kernel_initializer=HeNormal(seed=seed + 3),
                                 kernel_regularizer=l1_l2(l1=hp.Float('l1_1', 1e-6, 1e-3, sampling='log'),
                                                          l2=hp.Float('l2_1', 1e-6, 1e-3, sampling='log')))))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)))

    model.add(TimeDistributed(Dense(input_shape[1], kernel_initializer=HeNormal(seed=seed + 4))))
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')), loss='mse')
    return model

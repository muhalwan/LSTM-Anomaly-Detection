import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model_definition_hp import build_lstm_autoencoder_hp


def tune_hyperparameters(X_train, input_shape, batch_size):
    tuner = kt.RandomSearch(
        lambda hp: build_lstm_autoencoder_hp(input_shape, hp),
        objective='val_loss',
        max_trials=3,
        executions_per_trial=1,
        directory='hp',
        project_name='rs'
    )
    stop_early = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    tuner.search(
        X_train, X_train,
        epochs=50,
        validation_split=0.2,
        batch_size=batch_size,
        callbacks=[stop_early, reduce_lr],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    return best_hps
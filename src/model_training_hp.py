from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model_definition_hp import build_lstm_autoencoder_hp


def train_lstm_autoencoder(X_train, hp, input_shape, epoch, batch_size):
    model = build_lstm_autoencoder_hp(input_shape, hp)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('model/best_model_hp.h5', save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, X_train,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        shuffle=True
    )

    return model, history

from utils.data_preprocessing import load_and_preprocess_data
from model_training_hyperparameter_tuning import tune_hyperparameters
from sklearn.model_selection import train_test_split


def main():

    try:
        X_sequences, original_data, scaler, ordinal_encoder, features = load_and_preprocess_data()

        # Split data into train and test sets
        X_train, X_test = train_test_split(X_sequences, test_size=0.2, random_state=42)

        input_shape = (X_train.shape[1], X_train.shape[2])

        # Tune hyperparameters using only the training data
        best_lstm_hps = tune_hyperparameters(X_train, input_shape, batch_size=32)

        print(type(best_lstm_hps))

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

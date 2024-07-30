import time
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from utils.data_preprocessing import load_and_preprocess_data
from model_training_hp import train_lstm_autoencoder
from utils.model_evaluation import (
    evaluate_lstm_autoencoder, evaluate_kmeans, plot_reconstruction_error_distribution,
    print_evaluation_results, plot_learning_curves, reshape_to_2d, plot_reconstruction_error_over_time
)
from model_training_hyperparameter_tuning import tune_hyperparameters
from main_tune_kmean import tune_kmeans
import os
from keras_tuner import HyperParameters

# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def main():
    try:
        X_sequences, original_data, scaler, encoder, features = load_and_preprocess_data()

        X_train, X_test = train_test_split(X_sequences, test_size=0.2, random_state=42)
        input_shape = (X_train.shape[1], X_train.shape[2])
        original_data_test = original_data.iloc[-len(X_test):]

        # Tune hyperparameters for LSTM
        best_lstm_hps = tune_hyperparameters(X_train, input_shape, batch_size=32)

        # LSTM Autoencoder
        start_time = time.time()
        lstm_model, history = train_lstm_autoencoder(X_train, best_lstm_hps, input_shape, epoch=10, batch_size=32)
        lstm_training_time = time.time() - start_time

        # Evaluate LSTM Autoencoder
        lstm_results, lstm_anomalies, lstm_re = evaluate_lstm_autoencoder(lstm_model, X_test,
                                                                          lstm_training_time, original_data_test)

        # Tune hyperparameters for K-means
        X_test_flattened = reshape_to_2d(X_test)
        best_kmeans_params = tune_kmeans(X_test_flattened)

        # K-means with tuned hyperparameters
        start_time = time.time()
        kmeans = KMeans(**best_kmeans_params, random_state=42)
        kmeans.fit(X_test_flattened)
        kmeans_training_time = time.time() - start_time

        # Evaluate K-means
        kmeans_results, kmeans_anomalies, kmeans_re = evaluate_kmeans(kmeans, X_test, kmeans_training_time,
                                                                      original_data_test)

        # Print and plot results
        test_timestamps = original_data['Timestamp'].iloc[-len(X_test):]
        print_evaluation_results(lstm_results, kmeans_results)
        plot_reconstruction_error_distribution(lstm_re, kmeans_re)
        plot_reconstruction_error_over_time(lstm_re, kmeans_re, test_timestamps)
        plot_learning_curves(history)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

import time
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from utils.data_preprocessing import load_and_preprocess_data
from model_training_base import train_lstm_model_base
from utils.model_evaluation import (
    evaluate_lstm_autoencoder, evaluate_kmeans, plot_reconstruction_error_distribution,
    print_evaluation_results, plot_learning_curves, reshape_to_2d, plot_reconstruction_error_over_time
)
import os


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def main(use_subset=False):
    try:
        X_sequences, original_data, scaler, encoder, features = load_and_preprocess_data()

        if use_subset:
            subset_size = 10000
            X_sequences = X_sequences[:subset_size]

        X_train, X_test = train_test_split(X_sequences, test_size=0.2, random_state=42)
        original_data_test = original_data.iloc[-len(X_test):]

        input_shape = (X_train.shape[1], X_train.shape[2])
        # LSTM Autoencoder
        start_time = time.time()
        lstm_model, history = train_lstm_model_base(X_train, input_shape, epoch=100, batch_size=32)
        lstm_training_time = time.time() - start_time

        # Evaluate LSTM Autoencoder
        lstm_results, lstm_anomalies, lstm_re = evaluate_lstm_autoencoder(lstm_model, X_test,
                                                                          lstm_training_time, original_data_test)

        # K-means
        start_time = time.time()
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        kmeans.fit(reshape_to_2d(X_test))
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

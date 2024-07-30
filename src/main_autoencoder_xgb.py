import time
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from utils.data_preprocessing import load_and_preprocess_data
from model_training_hpxg import train_lstm_autoencoder_hpxg
from utils.model_evaluation import (
    evaluate_lstm_autoencoder_xgb, evaluate_kmeans_xgb, plot_reconstruction_error_distribution,
    print_evaluation_results, plot_learning_curves, reshape_to_2d, plot_reconstruction_error_over_time
)
from model_training_hyperparameter_tuning import tune_hyperparameters
from main_tune_kmean import tune_kmeans
import os
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def main():
    try:
        X_sequences, original_data, scaler, encoder, features = load_and_preprocess_data()

        X_train, X_test = train_test_split(X_sequences, test_size=0.2, random_state=42)
        input_shape = (X_train.shape[1], X_train.shape[2])
        original_data_test = original_data.iloc[-len(X_test):]

        # XGB
        X_train_reshaped = reshape_to_2d(X_train)
        X_test_reshaped = reshape_to_2d(X_test)
        xgb_train = X_train[:, -1, 0]  # Use the last timestep of the first feature

        xgb_model = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train_reshaped, xgb_train)
        xgb_pred = xgb_model.predict(X_test_reshaped)

        # LSTM with hyperparameters and XGBoost
        best_lstm_hps = tune_hyperparameters(X_train, input_shape, batch_size=32)

        start_time = time.time()
        lstm_model, history = train_lstm_autoencoder_hpxg(X_train, best_lstm_hps, input_shape, epoch=50, batch_size=32)

        lstm_pred = lstm_model.predict(X_test)

        combined_pred_lstm = lstm_pred.copy()
        combined_pred_lstm[:, :, 0] = 0.7 * lstm_pred[:, :, 0] + 0.3 * xgb_pred.reshape(-1, 1)

        lstm_training_time = time.time() - start_time

        lstm_results, lstm_anomalies, lstm_re = evaluate_lstm_autoencoder_xgb(
            combined_pred_lstm, X_test, lstm_training_time, original_data_test
        )

        # K-means with hyperparameters and XGBoost
        X_test_flattened = reshape_to_2d(X_test)
        best_kmeans_params = tune_kmeans(X_test_flattened)

        start_time = time.time()
        kmeans = KMeans(**best_kmeans_params, random_state=42)
        kmeans.fit(X_test_flattened)

        kmeans_labels = kmeans.predict(X_test_flattened)
        kmeans_centroids = kmeans.cluster_centers_

        # Reshape kmeans_centroids to match X_test shape
        kmeans_pred = kmeans_centroids[kmeans_labels].reshape(X_test.shape)

        # Combine K-means and XGBoost predictions
        combined_pred_kmeans = kmeans_pred.copy()
        combined_pred_kmeans[:, :, 0] = 0.7 * kmeans_pred[:, :, 0] + 0.3 * xgb_pred.reshape(-1, 1)

        kmeans_training_time = time.time() - start_time

        kmeans_results, kmeans_anomalies, kmeans_re = evaluate_kmeans_xgb(
            combined_pred_kmeans, X_test, kmeans_training_time, original_data_test
        )

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

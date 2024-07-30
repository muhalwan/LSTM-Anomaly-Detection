import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from utils.data_preprocessing import load_and_preprocess_data
from utils.model_evaluation import (
    evaluate_lstm_autoencoder, evaluate_kmeans, plot_reconstruction_error_distribution,
    print_evaluation_results, reshape_to_2d, plot_reconstruction_error_over_time
)


def analyze_anomalies(anomaly_logs):
    out_of_hours = anomaly_logs[anomaly_logs['Is_Out_of_Hours'] == 1]
    weekend = anomaly_logs[anomaly_logs['Is_Weekend'] == 1]
    unusual_time = anomaly_logs[
        np.abs(anomaly_logs['Time_Since_Last_Log'] - anomaly_logs['Time_Since_Last_Log'].mean()) > 3 * anomaly_logs[
            'Time_Since_Last_Log'].std()]
    large_packets = anomaly_logs[anomaly_logs['Has_Large_Packet'] == 1]

    return {
        "Out of Hours": out_of_hours,
        "Weekend": weekend,
        "Unusual Time": unusual_time,
        "Large Packets": large_packets,
    }


def print_anomaly_details(anomaly_logs):
    print("\nDetailed Anomaly Analysis:")
    print(f"Total Anomalies Detected: {len(anomaly_logs)}")

    analyzed_anomalies = analyze_anomalies(anomaly_logs)

    for category, data in analyzed_anomalies.items():
        print(f"\n{category} Anomalies:")
        print(f"Count: {len(data)}")
        if len(data) > 0:
            print(data.head(3).to_string())  # Show only the first 3 entries
        else:
            print("No anomalies detected in this category.")


def main(model_path='model/best_model_base.h5'):
    try:
        # Load data
        X_sequences, original_data, scaler, encoder, features = load_and_preprocess_data()

        X_train, X_test = train_test_split(X_sequences, test_size=0.2, random_state=42)
        original_data_test = original_data.iloc[-len(X_test):]

        # Load pre-trained LSTM model
        print("Loading pre-trained LSTM model...")
        lstm_model = load_model(model_path)

        # LSTM Autoencoder evaluation
        print("Evaluating LSTM Autoencoder...")
        lstm_results, lstm_anomalies, lstm_re = evaluate_lstm_autoencoder(lstm_model, X_test, 0, original_data_test)

        # K-means
        print("Performing K-means clustering...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        kmeans.fit(reshape_to_2d(X_test))

        # Evaluate K-means
        print("Evaluating K-means...")
        kmeans_results, kmeans_anomalies, kmeans_re = evaluate_kmeans(kmeans, X_test, 0, original_data_test)

        # Detailed analysis of LSTM anomalies
        print("LSTM Anomalies")
        lstm_anomaly_indices = np.where(lstm_anomalies)[0]
        lstm_anomaly_logs = original_data_test.iloc[lstm_anomaly_indices]
        print_anomaly_details(lstm_anomaly_logs)

        # Detailed analysis of Kmeans anomalies
        print("KMeans Anomalies")
        kmeans_anomaly_indices = np.where(kmeans_anomalies)[0]
        kmeans_anomaly_logs = original_data_test.iloc[kmeans_anomaly_indices]
        print_anomaly_details(kmeans_anomaly_logs)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

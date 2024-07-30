import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, precision_recall_curve, auc, f1_score,
                             precision_score, recall_score,
                             r2_score, explained_variance_score)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest


def reshape_to_2d(X):
    """Reshape 3D array to 2D if necessary."""
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X


def dynamic_threshold(reconstruction_error, n_sigma=3):
    mean = np.mean(reconstruction_error)
    std = np.std(reconstruction_error)
    return mean + n_sigma * std


def plot_reconstruction_error(reconstruction_error, threshold, title):
    plt.figure(figsize=(12, 6))
    plt.hist(reconstruction_error, bins=50, density=True, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
    plt.title(title)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.savefig(f'image/3/{title.lower().replace(" ", "_")}.png')
    plt.show()
    plt.close()


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_lstm_autoencoder_xgb(X_test_pred, X_test, training_time, original_data_test):
    """Evaluate LSTM autoencoder model."""
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    X_test_pred_2d = X_test_pred.reshape(X_test_pred.shape[0], -1)

    test_mse = mean_squared_error(X_test_2d, X_test_pred_2d)
    test_rmse = rmse(X_test_2d, X_test_pred_2d)
    test_mae = mean_absolute_error(X_test_2d, X_test_pred_2d)
    test_r2 = r2_score(X_test_2d, X_test_pred_2d)
    test_evs = explained_variance_score(X_test_2d, X_test_pred_2d)

    reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))

    threshold = dynamic_threshold(reconstruction_error)
    anomalies = reconstruction_error > threshold

    temporal_anomalies = analyze_temporal_anomalies(original_data_test, anomalies)
    content_anomalies = analyze_content_anomalies(original_data_test, anomalies)
    statistical_anomalies = analyze_statistical_anomalies(original_data_test, anomalies)

    results = {
        "Training Time": training_time,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "MAE": test_mae,
        "R-squared": test_r2,
        "Explained Variance Score": test_evs,
        "Anomalies": np.sum(anomalies),
        "Anomaly Percentage": np.mean(anomalies) * 100,
        "Anomaly Threshold": threshold,
        "Temporal Anomalies": temporal_anomalies,
        "Content Anomalies": content_anomalies,
        "Statistical Anomalies": statistical_anomalies
    }

    plot_reconstruction_error(reconstruction_error, threshold, 'LSTM Reconstruction Error')

    return results, anomalies, reconstruction_error


def evaluate_kmeans_xgb(X_test_pred, X_test, training_time, original_data):
    """Evaluate K-means clustering model."""
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    X_test_pred_2d = X_test_pred.reshape(X_test_pred.shape[0], -1)

    reconstruction_error = np.sum(np.square(X_test_2d - X_test_pred_2d), axis=1)

    mse = mean_squared_error(X_test_2d, X_test_pred_2d)
    rmse_value = rmse(X_test_2d, X_test_pred_2d)
    mae = mean_absolute_error(X_test_2d, X_test_pred_2d)
    r2 = r2_score(X_test_2d, X_test_pred_2d)
    evs = explained_variance_score(X_test_2d, X_test_pred_2d)

    threshold = dynamic_threshold(reconstruction_error)
    anomalies = reconstruction_error > threshold

    temporal_anomalies = analyze_temporal_anomalies(original_data, anomalies)
    content_anomalies = analyze_content_anomalies(original_data, anomalies)
    statistical_anomalies = analyze_statistical_anomalies(original_data, anomalies)

    results = {
        "Training Time": training_time,
        "MSE": mse,
        "RMSE": rmse_value,
        "MAE": mae,
        "R-squared": r2,
        "Explained Variance Score": evs,
        "Anomalies": np.sum(anomalies),
        "Anomaly Percentage": np.mean(anomalies) * 100,
        "Anomaly Threshold": threshold,
        "Temporal Anomalies": temporal_anomalies,
        "Content Anomalies": content_anomalies,
        "Statistical Anomalies": statistical_anomalies
    }

    plot_reconstruction_error(reconstruction_error, threshold, 'K-means Reconstruction Error')

    return results, anomalies, reconstruction_error


def evaluate_lstm_autoencoder(model, X_test, training_time, original_data_test):
    """Evaluate LSTM autoencoder model."""
    X_test_pred = model.predict(X_test)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    X_test_pred_2d = X_test_pred.reshape(X_test_pred.shape[0], -1)

    test_mse = mean_squared_error(X_test_2d, X_test_pred_2d)
    test_rmse = rmse(X_test_2d, X_test_pred_2d)
    test_mae = mean_absolute_error(X_test_2d, X_test_pred_2d)
    test_r2 = r2_score(X_test_2d, X_test_pred_2d)
    test_evs = explained_variance_score(X_test_2d, X_test_pred_2d)

    reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))

    threshold = dynamic_threshold(reconstruction_error)
    anomalies = reconstruction_error > threshold

    temporal_anomalies = analyze_temporal_anomalies(original_data_test, anomalies)
    content_anomalies = analyze_content_anomalies(original_data_test, anomalies)
    statistical_anomalies = analyze_statistical_anomalies(original_data_test, anomalies)

    results = {
        "Training Time": training_time,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "MAE": test_mae,
        "R-squared": test_r2,
        "Explained Variance Score": test_evs,
        "Anomalies": np.sum(anomalies),
        "Anomaly Percentage": np.mean(anomalies) * 100,
        "Anomaly Threshold": threshold,
        "Temporal Anomalies": temporal_anomalies,
        "Content Anomalies": content_anomalies,
        "Statistical Anomalies": statistical_anomalies
    }

    plot_reconstruction_error(reconstruction_error, threshold, 'LSTM Reconstruction Error')

    return results, anomalies, reconstruction_error


def evaluate_kmeans(kmeans, X, training_time, original_data):
    """Evaluate K-means clustering model."""
    X_2d = X.reshape(X.shape[0], -1)
    kmeans_labels = kmeans.predict(X_2d)
    kmeans_centroids = kmeans.cluster_centers_

    reconstruction_error = np.sum(np.square(X_2d - kmeans_centroids[kmeans_labels]), axis=1)

    mse = mean_squared_error(X_2d, kmeans_centroids[kmeans_labels])
    rmse_value = rmse(X_2d, kmeans_centroids[kmeans_labels])
    mae = mean_absolute_error(X_2d, kmeans_centroids[kmeans_labels])
    r2 = r2_score(X_2d, kmeans_centroids[kmeans_labels])
    evs = explained_variance_score(X_2d, kmeans_centroids[kmeans_labels])

    threshold = dynamic_threshold(reconstruction_error)
    anomalies = reconstruction_error > threshold

    temporal_anomalies = analyze_temporal_anomalies(original_data, anomalies)
    content_anomalies = analyze_content_anomalies(original_data, anomalies)
    statistical_anomalies = analyze_statistical_anomalies(original_data, anomalies)

    results = {
        "Training Time": training_time,
        "MSE": mse,
        "RMSE": rmse_value,
        "MAE": mae,
        "R-squared": r2,
        "Explained Variance Score": evs,
        "Anomalies": np.sum(anomalies),
        "Anomaly Percentage": np.mean(anomalies) * 100,
        "Anomaly Threshold": threshold,
        "Temporal Anomalies": temporal_anomalies,
        "Content Anomalies": content_anomalies,
        "Statistical Anomalies": statistical_anomalies
    }

    plot_reconstruction_error(reconstruction_error, threshold, 'K-means Reconstruction Error')

    return results, anomalies, reconstruction_error


def analyze_temporal_anomalies(data, anomalies):
    """Analyze temporal anomalies."""
    temporal_anomalies = {
        "Out of Hours": np.sum(data['Is_Out_of_Hours'].values[anomalies] == 1),
        "Weekend": np.sum(data['Is_Weekend'].values[anomalies] == 1),
        "Unusual Time Since Last Log": np.sum(zscore(data['Time_Since_Last_Log'].values[anomalies]) > 3)
    }
    return temporal_anomalies


def analyze_content_anomalies(data, anomalies):
    """Analyze content-based anomalies."""
    content_anomalies = {
        "Unusual Message Length": np.sum(np.abs(zscore(data['Message_Length'].values[anomalies])) > 3),
        "Large Packets": np.sum(data['Has_Large_Message'].values[anomalies] == 1)
    }
    return content_anomalies


def analyze_statistical_anomalies(data, anomalies):
    """Analyze statistical anomalies."""
    statistical_anomalies = {
        "High Log Frequency": np.sum(data['Log_Frequency_Z_Score'].values[anomalies] > 3),
        "Low Log Frequency": np.sum(data['Log_Frequency_Z_Score'].values[anomalies] < -3),
        "High Thread Activity": np.sum(data['Is_High_Frequency_Thread'].values[anomalies] == 1)
    }
    return statistical_anomalies


def print_evaluation_results(lstm_results, kmeans_results):
    """Print evaluation results for both models."""
    print("\nLSTM Autoencoder Results:")
    for key, value in lstm_results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

    print("\nK-means Clustering Results:")
    for key, value in kmeans_results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


def plot_reconstruction_error_distribution(lstm_re, kmeans_re):
    """Plot reconstruction error distribution for LSTM and K-means."""
    plt.figure(figsize=(12, 6))
    sns.histplot(lstm_re, kde=True, label='LSTM', stat='density', alpha=0.6)
    sns.histplot(kmeans_re, kde=True, label='K-means', stat='density', alpha=0.6)
    plt.title('Reconstruction Error Distribution: LSTM vs K-means')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.xscale('log')
    plt.savefig('image/3/reconstruction_error_distribution.png')
    plt.show()
    plt.close()


def plot_reconstruction_error_over_time(lstm_re, kmeans_re, timestamps):
    """Plot reconstruction error over time for LSTM and K-means."""
    plt.figure(figsize=(15, 8))
    plt.plot(timestamps, lstm_re, label='LSTM', alpha=0.7)
    plt.plot(timestamps, kmeans_re, label='K-means', alpha=0.7)
    plt.title('Reconstruction Error Over Time: LSTM vs K-means')
    plt.xlabel('Time')
    plt.ylabel('Reconstruction Error (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('image/3/reconstruction_error_over_time.png')
    plt.show()
    plt.close()


def plot_learning_curves(history):
    """Plot learning curves from model training history."""
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('image/3/learning_curves.png')
    plt.show()
    plt.close()

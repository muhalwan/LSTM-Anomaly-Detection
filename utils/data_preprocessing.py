import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


def create_features(df):
    df = df.copy()

    # Temporal features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='ISO8601')
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Week_of_Year'] = df['Timestamp'].dt.isocalendar().week
    df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
    df['Time_of_Day'] = pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                               include_lowest=True)
    df['Time_Since_Last_Log'] = df['Timestamp'].diff().dt.total_seconds().fillna(0)
    business_hours = (df['Hour'] >= 9) & (df['Hour'] < 17) & (df['Is_Weekend'] == 0)
    df['Is_Out_of_Hours'] = (~business_hours).astype(int)

    # Log frequency features
    df['Log_Frequency'] = df.groupby(['Hour', 'Day_of_Week'])['Timestamp'].transform('count')
    df['Rolling_Mean_Log_Frequency'] = df.groupby('Day_of_Week')['Log_Frequency'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean())
    df['Rolling_Std_Log_Frequency'] = df.groupby('Day_of_Week')['Log_Frequency'].transform(
        lambda x: x.rolling(window=10, min_periods=1).std())
    df['Log_Frequency_Z_Score'] = (df['Log_Frequency'] - df['Rolling_Mean_Log_Frequency']) / df[
        'Rolling_Std_Log_Frequency']

    # Message content features
    df['Message_Length'] = df['Message'].str.len()
    df['Has_Error'] = df['Message'].str.contains('error|fail|critical', case=False).astype(int)
    df['Has_Network_Issue'] = df['Message'].str.contains('network|connection|timeout', case=False).astype(int)
    df['Has_Large_Message'] = (df['Message_Length'] > df['Message_Length'].quantile(0.95)).astype(int)

    # Thread ID features
    df['Thread_ID_Frequency'] = df.groupby('Thread_ID')['Timestamp'].transform('count')
    df['Is_High_Frequency_Thread'] = (df['Thread_ID_Frequency'] > df['Thread_ID_Frequency'].quantile(0.95)).astype(int)

    # Interaction terms
    df['Hour_Day_Interaction'] = df['Hour'] * df['Day_of_Week']

    return df


def extract_text_features(messages, max_features=100):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(messages)
    return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())


def load_and_preprocess_data():
    data_path = "../data/ovs-vswitchd.10.csv"
    sequence_length = 100
    data = pd.read_csv(data_path)
    print("Data loaded successfully.")

    data = create_features(data)

    numeric_features = ['Hour', 'Day_of_Week', 'Month', 'Week_of_Year', 'Is_Weekend', 'Time_Since_Last_Log',
                        'Is_Out_of_Hours', 'Log_Frequency', 'Rolling_Mean_Log_Frequency', 'Rolling_Std_Log_Frequency',
                        'Log_Frequency_Z_Score', 'Message_Length', 'Thread_ID_Frequency', 'Is_High_Frequency_Thread',
                        'Hour_Day_Interaction']
    categorical_features = ['Time_of_Day']

    # Handle missing values for numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    X_numeric = numeric_imputer.fit_transform(data[numeric_features])

    # Encode categorical features
    ordinal_encoder = OrdinalEncoder()
    X_categorical = ordinal_encoder.fit_transform(data[categorical_features])

    # Combine all features
    X = np.hstack((X_numeric, X_categorical))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences for LSTM
    X_sequences = np.array([X_scaled[i:i + sequence_length] for i in range(len(X_scaled) - sequence_length + 1)])

    # Check for NaN or infinite values
    if np.isnan(X_sequences).any() or np.isinf(X_sequences).any():
        print("Warning: NaN or infinite values found in sequences after creating sequences")
        print(f"NaN values: {np.isnan(X_sequences).sum()}")
        print(f"Infinite values: {np.isinf(X_sequences).sum()}")

        # Replace any remaining NaN or inf values
        X_sequences = np.nan_to_num(X_sequences, nan=np.float32(0), posinf=np.float32(np.finfo(np.float32).max),
                                    neginf=np.float32(np.finfo(np.float32).min))
    else:
        print("No NaN or infinite values found in sequences.")

    print("Data preprocessing completed successfully.")
    print(f"X_sequences shape: {X_sequences.shape}")

    return X_sequences, data, scaler, ordinal_encoder, numeric_features + categorical_features

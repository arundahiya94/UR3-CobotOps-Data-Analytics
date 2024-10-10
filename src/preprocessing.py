import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

def load_data(file_path):
    return pd.read_excel(file_path)

def clean_data(df):
    # Convert columns to float
    df['grip_lost'] = df['grip_lost'].astype(float)
    df['Robot_ProtectiveStop'] = df['Robot_ProtectiveStop'].astype(float)

    # Drop missing values
    df.dropna(inplace=True)

    # Drop unnecessary columns
    df.drop(columns=['Timestamp', 'Num'], inplace=True)

    return df

def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def apply_power_transform(df, columns):
    pt = PowerTransformer(method='yeo-johnson')
    df[columns] = pt.fit_transform(df[columns])
    return df

def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1

    for col in columns:
        lower_bound = Q1[col] - 1.5 * IQR[col]
        upper_bound = Q3[col] + 1.5 * IQR[col]
        df[col] = df[col].apply(lambda x: max(min(x, upper_bound), lower_bound))

    return df

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_data(input_path, target_column, output_dataset_path, pipeline_path, column_path):

    # Load data
    df = pd.read_csv(input_path)

    # Identifikasi kolom numerik
    numeric_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)

    # Simpan nama kolom
    pd.DataFrame(columns=df.drop(columns=[target_column]).columns)\
        .to_csv(column_path, index=False)

    # Handling missing value
    kolom_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[kolom_missing] = df[kolom_missing].replace(0, np.nan)

    for col in kolom_missing:
        df[col].fillna(df[col].median(), inplace=True)

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline scaling
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )

    # Transform data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Konversi ke DataFrame
    feature_names = numeric_features + \
        [col for col in X.columns if col not in numeric_features]

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)

    # Gabungkan dengan target
    final_df = pd.concat(
        [X_train_df.reset_index(drop=True), y_train.reset_index(drop=True)],
        axis=1
    )

    # Save dataset hasil preprocessing
    final_df.to_csv(output_dataset_path, index=False)
    print(f"Dataset hasil preprocessing disimpan di: {output_dataset_path}")

    # Save pipeline
    dump(preprocessor, pipeline_path)
    print(f"Pipeline disimpan di: {pipeline_path}")


if __name__ == "__main__":
    preprocess_data(
        input_path="diabetes_raw.csv",
        target_column="Outcome",
        output_dataset_path="preprocessing/diabetes_preprocessing/diabetes_clean.csv",
        pipeline_path="preprocessing/diabetes_preprocessing/diabetes_preprocessor.joblib",
        column_path="preprocessing/diabetes_preprocessing/columns.csv"
    )
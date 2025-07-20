import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

DATA_PATH = "adult 3.csv"
MODEL_PATH = "salary_mlp_pipeline.pkl"
FEATURE_PATH = "salary_mlp_features.pkl"
TARGET_PATH = "salary_mlp_target.pkl"
ENCODER_PATH = "salary_mlp_target_encoder.pkl"

st.set_page_config(page_title="Salary Prediction App", layout="wide")
st.title("💼 Employee Salary Prediction App")
st.write("MLPClassifier (Balanced) with Confidence Visualization")

@st.cache_resource
def load_model():
    if Path(MODEL_PATH).exists():
        pipe = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURE_PATH)
        target_col = joblib.load(TARGET_PATH)
        y_le = joblib.load(ENCODER_PATH) if Path(ENCODER_PATH).exists() else None
        data = pd.read_csv(DATA_PATH)
        return pipe, features, target_col, y_le, data
    else:
        # Train if no saved model
        data = pd.read_csv(DATA_PATH)
        target_candidates = [c for c in ["salary","income","Salary","Income"] if c in data.columns]
        target_col = target_candidates[0] if target_candidates else data.columns[-1]
        features = [c for c in ['age','educational-num','marital-status','occupation','gender','workclass'] if c in data.columns]

        X = data[features].copy()
        y_raw = data[target_col]
        y_le = None
        if y_raw.dtype == 'object':
            y_le = LabelEncoder()
            y = y_le.fit_transform(y_raw)
        else:
            y = y_raw

        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]

        X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        preproc = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])
        mlp = MLPClassifier(
            hidden_layer_sizes=(256,128,64),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=2000,
            alpha=1e-5,
            random_state=42,
            class_weight='balanced'
        )
        pipe = Pipeline([("prep", preproc), ("clf", mlp)])
        pipe.fit(X_train_df, y_train)

        joblib.dump(pipe, MODEL_PATH)
        joblib.dump(features, FEATURE_PATH)
        joblib.dump(target_col, TARGET_PATH)
        if y_le:
            joblib.dump(y_le, ENCODER_PATH)
        return pipe, features, target_col, y_le, data

pipe, features, target_col, y_le, data = load_model()

st.subheader("Enter Employee Details:")
inputs = {}
for col in features:
    if pd.api.types.is_numeric_dtype(data[col]):
        inputs[col] = st.number_input(
            col, min_value=int(data[col].min()),
            max_value=int(data[col].max()),
            value=int(data[col].median())
        )
    else:
        inputs[col] = st.selectbox(col, sorted(data[col].dropna().unique().tolist()))

if st.button("Predict Salary"):
    input_df = pd.DataFrame([inputs])
    prediction = pipe.predict(input_df)
    if y_le:
        prediction = y_le.inverse_transform(prediction)
    st.success(f"Predicted {target_col}: {prediction[0]}")

    # Show probabilities
    if hasattr(pipe.named_steps['clf'], "predict_proba"):
        proba = pipe.predict_proba(input_df)[0]
        classes = y_le.classes_ if y_le else pipe.named_steps['clf'].classes_
        st.subheader("Prediction Confidence")
        st.write({cls: round(p*100, 2) for cls, p in zip(classes, proba)})

        # Plot bar chart
        fig, ax = plt.subplots()
        ax.bar(classes, proba, color=['#4caf50', '#2196f3'])
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Confidence')
        st.pyplot(fig)

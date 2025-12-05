import re
import sys
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pathlib import Path

# DATA_PATH = "/home/zooboo/Documents/school/ai/chatbot2/dataset.csv"
DATA_PATH = Path(__file__).parent / "dataset.csv"

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the medical dataset.
    Expects at least these columns:
      Disease, Fever, Cough, Fatigue, Difficulty Breathing,
      Age, Gender, Blood Pressure, Cholesterol Level
    """
    df = pd.read_csv(path)
    required_cols = [
        "Disease",
        "Fever",
        "Cough",
        "Fatigue",
        "Difficulty Breathing",
        "Age",
        "Gender",
        "Blood Pressure",
        "Cholesterol Level",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[required_cols].copy()


def train_model(df: pd.DataFrame) -> Pipeline:
    """
    Train a simple classifier that predicts Disease from symptoms and demographics.
    """
    feature_cols = [
        "Fever",
        "Cough",
        "Fatigue",
        "Difficulty Breathing",
        "Age",
        "Gender",
        "Blood Pressure",
        "Cholesterol Level",
    ]
    X = df[feature_cols]
    y = df["Disease"]

    numeric_features = ["Age"]
    categorical_features = [
        "Fever",
        "Cough",
        "Fatigue",
        "Difficulty Breathing",
        "Gender",
        "Blood Pressure",
        "Cholesterol Level",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    model.fit(X, y)
    return model


def parse_free_text_description(text: str) -> Dict[str, Any]:
    """
    Very simple NLP to pull out symptoms and a bit of demographic info
    from a sentence-like description.

    Example inputs:
      "I'm a 25 year old male with a fever and bad cough, very tired."
      "Female, 30, no fever but I have difficulty breathing and feel exhausted."
    """
    info: Dict[str, Any] = {
        "Fever": None,
        "Cough": None,
        "Fatigue": None,
        "Difficulty Breathing": None,
        "Age": None,
        "Gender": None,
        "Blood Pressure": None,
        "Cholesterol Level": None,
    }

    text_lower = text.lower()

    # Fever
    if "fever" in text_lower or "temperature" in text_lower:
        info["Fever"] = "Yes"
    if any(p in text_lower for p in ["no fever", "without fever"]):
        info["Fever"] = "No"

    # Cough
    if "cough" in text_lower:
        info["Cough"] = "Yes"
    if any(p in text_lower for p in ["no cough", "without cough"]):
        info["Cough"] = "No"

    # Fatigue / tiredness
    if any(w in text_lower for w in ["tired", "fatigue", "exhausted", "weakness"]):
        info["Fatigue"] = "Yes"
    if any(p in text_lower for p in ["not tired", "no fatigue"]):
        info["Fatigue"] = "No"

    # Difficulty breathing
    if any(
        w in text_lower
        for w in [
            "difficulty breathing",
            "shortness of breath",
            "short of breath",
            "hard to breathe",
            "can't breathe",
            "cannot breathe",
        ]
    ):
        info["Difficulty Breathing"] = "Yes"
    if any(p in text_lower for p in ["no trouble breathing", "no breathing problems"]):
        info["Difficulty Breathing"] = "No"

    # Age (take the first reasonable number mentioned)
    age_match = re.search(
        r"(\d+)\s*(?:years?\s*old|year-old|yo|yrs?|y/o)?", text_lower
    )
    if age_match:
        try:
            age_val = int(age_match.group(1))
            if 0 < age_val < 120:  # simple sanity check
                info["Age"] = age_val
        except ValueError:
            pass

    # Gender
    if any(w in text_lower for w in ["female", "woman", "girl", "feminine"]):
        info["Gender"] = "Female"
    if any(w in text_lower for w in ["male", "man", "boy", "masculine"]):
        info["Gender"] = "Male"

    # Blood pressure (very rough keywords)
    if "high blood pressure" in text_lower or "hypertension" in text_lower:
        info["Blood Pressure"] = "High"
    if "low blood pressure" in text_lower or "hypotension" in text_lower:
        info["Blood Pressure"] = "Low"
    # Generic "normal blood pressure"
    if "normal blood pressure" in text_lower:
        info["Blood Pressure"] = "Normal"

    # Cholesterol level (rough keywords)
    if "high cholesterol" in text_lower:
        info["Cholesterol Level"] = "High"
    if "low cholesterol" in text_lower:
        info["Cholesterol Level"] = "Low"
    if "normal cholesterol" in text_lower:
        info["Cholesterol Level"] = "Normal"

    return info


def fill_missing_with_mode_and_mean(
    info: Dict[str, Any], df: pd.DataFrame
) -> Dict[str, Any]:
    """
    For any missing features, fill with the most common value (categorical)
    or mean (numeric) from the dataset.
    This lets the model still make a prediction when the user skips questions.
    """
    completed = dict(info)
    for col in ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender", "Blood Pressure", "Cholesterol Level"]:
        if completed.get(col) is None:
            completed[col] = df[col].mode().iloc[0]
    if completed.get("Age") is None:
        completed["Age"] = int(round(df["Age"].mean()))
    return completed


def predict_diagnosis(
    model: Pipeline, info: Dict[str, Any], df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Use the trained model to predict a disease and return probabilities.
    """
    feature_cols = [
        "Fever",
        "Cough",
        "Fatigue",
        "Difficulty Breathing",
        "Age",
        "Gender",
        "Blood Pressure",
        "Cholesterol Level",
    ]

    filled = fill_missing_with_mode_and_mean(info, df)
    X_input = pd.DataFrame([{col: filled[col] for col in feature_cols}])

    proba = model.predict_proba(X_input)[0]
    classes = model.classes_

    # Rank diseases by probability
    ranked_indices = np.argsort(proba)[::-1]
    best_idx = ranked_indices[0]
    best_disease = classes[best_idx]
    best_prob = float(proba[best_idx])

    # Grab top 3 suggestions
    top_suggestions = [
        (classes[i], float(proba[i]))
        for i in ranked_indices[:3]
    ]

    return {
        "best_disease": best_disease,
        "best_prob": best_prob,
        "top_suggestions": top_suggestions,
        "used_features": X_input.to_dict(orient="records")[0],
    }


def run_chatbot() -> None:
    print("Welcome to the Symptom Checker Chatbot.")
    print("This tool is for educational purposes only and is NOT medical advice.")
    print("Always consult a qualified healthcare professional for real medical concerns.\n")

    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print("Training internal model from dataset... (this may take a moment)")
    model = train_model(df)
    print("Model training complete.\n")

    while True:
        print("=" * 60)
        print("Describe your symptoms in one or two sentences.")
        print(
            "For example: 'I am a 25 year old male with a high fever, bad cough, and I'm very tired.'"
        )
        user_text = input("\nYour description (or type 'quit' to exit): ").strip()
        if user_text.lower() in ("quit", "exit"):
            print("Thank you for using the Symptom Checker Chatbot. Take care!")
            break

        info = parse_free_text_description(user_text)

        result = predict_diagnosis(model, info, df)

        print("\n--- Possible Diagnosis (Not Medical Advice) ---")
        best = result["best_disease"]
        conf = result["best_prob"]
        print(f"Most likely condition: {best} (confidence ~ {conf * 100:.1f}%)")
        print("\nOther possible conditions:")
        for disease, p in result["top_suggestions"]:
            print(f"  - {disease}: {p * 100:.1f}%")

        print(
            "\nPlease remember: this is a class project demo and not a substitute for a doctor."
        )

        again = input("\nWould you like to enter another set of symptoms? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("Thank you for using the Symptom Checker Chatbot. Take care!")
            break


if __name__ == "__main__":
    run_chatbot()



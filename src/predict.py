import pandas as pd
import joblib
import pathlib

# Resolve the project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Load the saved model
model_path = PROJECT_ROOT / "model" / "best_model.pkl"
model = joblib.load(model_path)

# Example new data
new_person = pd.DataFrame({
    'age': [35],
    'income': [70000],
    'gender': ['Female'],
    'education_level': ['PhD'],
    'marital_status': ['Single'],
    'children': [1],
    'job_type': ['Professional'],
    'previous_purchase': [0],
    'credit_score': [850],
    'interested_in_newsletter': [False]
})

# Predict
prediction = model.predict(new_person)
print("\nPrediction Result")
print("========================")
print(f"Will the person purchase? → {'Yes' if prediction[0] == 1 else 'No'}")
print("========================\n")

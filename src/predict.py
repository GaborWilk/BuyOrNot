import pandas as pd
import joblib
import pathlib

# Resolve the project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Load the saved model
model_path = PROJECT_ROOT / "model" / "best_pipeline.pkl"
model = joblib.load(model_path)

# Example new data
new_customer = pd.DataFrame({
    'age': [35],
    'income': [75000],
    'gender': ['Female']
})

# Predict
prediction = model.predict(new_customer)
print("Prediction:", prediction[0])

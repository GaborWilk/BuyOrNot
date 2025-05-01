import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pathlib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Resolve the project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def save_with_check(path, save_func, description="file"):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_func(path)
    if path.exists():
        print(f"{description.capitalize()} saved to: {path}")
    else:
        print(f"Failed to save {description} to: {path}")


# Generate realistic dataset
np.random.seed(42)
num_samples = 500

ages = np.random.randint(18, 70, size=num_samples)
incomes = np.random.normal(loc=60000, scale=15000, size=num_samples).astype(int)
genders = np.random.choice(['Male', 'Female'], size=num_samples)

# Simulate purchase behavior + randomness
purchase_prob = ((ages > 30) & (ages < 60) & (incomes > 55000)).astype(int)
noise = np.random.binomial(1, 0.2, size=num_samples)
purchased = np.clip(purchase_prob + noise, 0, 1)

data = pd.DataFrame({
    'age': ages,
    'income': incomes,
    'gender': genders,
    'purchased': purchased
})

# Save dataset to CSV (optional)
csv_path = PROJECT_ROOT / "data" / "realistic_purchases.csv"
save_with_check(
    path=csv_path,
    save_func=lambda p: data.to_csv(p, index=False),
    description="dataset"
)

# Split features and target
X = data.drop('purchased', axis=1)
y = data['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = ['age', 'income']
categorical_features = ['gender']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate
print("Best Parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=grid_search.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Save model
model_path = PROJECT_ROOT / "model" / "best_pipeline.pkl"
save_with_check(
    path=model_path,
    save_func=lambda p: joblib.dump(grid_search.best_estimator_, p),
    description="model"
)

# Feature importance
rf = grid_search.best_estimator_.named_steps['classifier']
preprocessor = grid_search.best_estimator_.named_steps['preprocessor']
ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_features = ohe.get_feature_names_out(['gender'])
all_features = numeric_features + list(cat_features)

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importance")
plt.title("Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

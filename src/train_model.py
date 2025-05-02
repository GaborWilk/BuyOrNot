import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from generate_data import generate_data

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
data = generate_data()

# Save dataset to CSV
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
numeric_features = ['age', 'income', 'children', 'previous_purchase', 'credit_score', 'interested_in_newsletter']
categorical_features = ['gender', 'education_level', 'marital_status', 'job_type']

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

# Models to test
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVC': SVC(probability=True, random_state=42)
}

results = {}
conf_matrices = {}
feature_importances = {}

# Train and evaluate
for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = {'accuracy': acc, 'y_pred': y_pred, 'model': pipe}

    # Confusion Matrix
    conf_matrices[name] = confusion_matrix(y_test, y_pred, labels=pipe.classes_)

    # Feature importance if supported
    if hasattr(model, 'feature_importances_'):
        ohe = pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        cat_names = ohe.get_feature_names_out(categorical_features)
        all_features = numeric_features + list(cat_names)
        feature_importances[name] = pd.Series(
            model.feature_importances_, index=all_features
        ).sort_values(ascending=False)

# Plot confusion matrices
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
axes_cm = axes_cm.flatten()
for i, (name, cm) in enumerate(conf_matrices.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=models[name].classes_)
    disp.plot(ax=axes_cm[i], cmap='Blues', colorbar=False)
    axes_cm[i].set_title(f"{name} Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot feature importances
fig_fi, axes_fi = plt.subplots(1, len(feature_importances), figsize=(15, 6))
if len(feature_importances) == 1:
    axes_fi = [axes_fi]

for ax, (name, importances) in zip(axes_fi, feature_importances.items()):
    importances.head(10).plot(kind='barh', ax=ax, color='teal')
    ax.set_title(f"{name} Top Features")
    ax.invert_yaxis()
plt.tight_layout()
plt.show()

# --- 7. Final accuracy report ---
print("\nFinal Model Accuracies:")
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
for name, res in results.items():
    print(f"{name}: Accuracy = {res['accuracy']:.4f}")

print(f"\nBest Model: {best_model[0]} with Accuracy = {best_model[1]['accuracy']:.4f}")


"""
# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# Hyperparameter tuning for RandomizedSearchCV
param_rand = {
    'classifier__n_estimators': [np.random.randint(50, 300)],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [np.random.randint(2, 11)],
    'classifier__min_samples_leaf': [np.random.randint(1, 5)],
    'classifier__max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    pipeline,
    param_rand,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
random_search.fit(X_train, y_train)

# Evaluate
print("\nBest Parameters for GridSearchCV:", grid_search.best_params_)
y_pred_gscv = grid_search.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred_gscv, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred_gscv))

print("\nBest Parameters for RandomizedSearchCV:", random_search.best_params_)
y_pred_rscv = random_search.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred_rscv, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred_rscv))

# Set up subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

# Subplot 1: Confusion Matrix GridSearchCV
cm = confusion_matrix(y_test, y_pred_gscv, labels=grid_search.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
disp.plot(cmap='Blues', ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix GSCV")

# Subplot 2: Feature Importance GridSearchCV
rf = grid_search.best_estimator_.named_steps['classifier']
preprocessor = grid_search.best_estimator_.named_steps['preprocessor']
ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_features = ohe.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_features)

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

axes[1].barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
axes[1].set_title("Feature Importances GSCV")
axes[1].set_xlabel("Importance")
axes[1].invert_yaxis()

# Subplot 3: Confusion Matrix RandomizedSearchCV
cm = confusion_matrix(y_test, y_pred_rscv, labels=random_search.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=random_search.classes_)
disp.plot(cmap='Blues', ax=axes[1], colorbar=False)
axes[1].set_title("Confusion Matrix RSCV")

# Subplot 4: Feature Importance RandomizedSearchCV
rf = random_search.best_estimator_.named_steps['classifier']
preprocessor = random_search.best_estimator_.named_steps['preprocessor']
ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_features = ohe.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_features)

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

axes[2].barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
axes[2].set_title("Feature Importances RSCV")
axes[2].set_xlabel("Importance")
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()
"""

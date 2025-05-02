import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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

# Define model specific param grids
param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 'log2']
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    },
    'LogisticRegression': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__solver': ['liblinear', 'lbfgs'],
        'classifier__penalty': ['l2']
    },
    'SVC': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }
}

results = {}
conf_matrices = {}
feature_importances = {}

# Train and evaluate
for name, model in models.items():
    print(f"Tuning and training: {name}")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    grid = GridSearchCV(
        pipe,
        param_grid=param_grids[name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model_pipeline = grid.best_estimator_
    y_pred = best_model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results[name] = {
        'accuracy': acc,
        'y_pred': y_pred,
        'model': best_model_pipeline,
        'best_params': grid.best_params_
    }

    # Confusion Matrix
    clf = best_model_pipeline.named_steps['classifier']
    conf_matrices[name] = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    # Feature importance if supported
    if hasattr(clf, 'feature_importances_'):
        fitted_preprocessor = best_model_pipeline.named_steps['preprocessor']
        ohe = fitted_preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_names = ohe.get_feature_names_out(categorical_features)
        all_features = numeric_features + list(cat_names)
        feature_importances[name] = pd.Series(
            clf.feature_importances_, index=all_features
        ).sort_values(ascending=False)

# Plot confusion matrices
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
axes_cm = axes_cm.flatten()
for i, (name, cm) in enumerate(conf_matrices.items()):
    clf = results[name]['model'].named_steps['classifier']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(ax=axes_cm[i], cmap='Blues', colorbar=False)
    axes_cm[i].set_title(f"{name} Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot feature importances only if there are any
if feature_importances:
    fig_fi, axes_fi = plt.subplots(1, len(feature_importances), figsize=(15, 6))
    if len(feature_importances) == 1:
        axes_fi = [axes_fi]

    for ax, (name, importances) in zip(axes_fi, feature_importances.items()):
        importances.head(10).plot(kind='barh', ax=ax, color='teal')
        ax.set_title(f"{name} Top Features")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

# Final accuracy report
print("\nFinal Model Accuracies:")
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
for name, res in results.items():
    print(f"{name}: Accuracy = {res['accuracy']:.4f}")

print(f"\nBest Model: {best_model[0]} with Accuracy = {best_model[1]['accuracy']:.4f}\n")

# Save best model
best_model_name, best_model_data = best_model
best_model_pipeline = best_model_data['model']
model_path = PROJECT_ROOT / "model" / "best_model.pkl"
save_with_check(
    path=model_path,
    save_func=lambda p: joblib.dump(best_model_pipeline, p),
    description=f"{best_model_name} model"
)

# BuyOrNot

## Predicting Customer Purchase Behavior with Machine Learning Pipelines

**BuyOrNot** is a machine learning project that predicts whether a customer will make a purchase (it's generic, it could be a product, house, etc.) based on their demographics, using scikit-learn pipelines, preprocessing, and hyperparameter tuning. It is a binary classification problem, and basically the project gives an
answer to this question:
    Based on age, income, and gender, will this person make a purchase?

Where the output (purchased column) can be the following:
- 1 = the person purchased (something)
- 0 = the person did not purchase

It's designed for learning, experimentation, and potential real-world adaptation.

### Results

The generated purchase data contains almost equal amounts of yes and no values.
![Generated Data](https://github.com/GaborWilk/BuyOrNot/blob/main/data/purchase_prob_before_noise.png?raw=true)

Final Model Accuracies:
- RandomForest: Accuracy = 0.7650
- GradientBoosting: Accuracy = 0.7550
- LogisticRegression: Accuracy = 0.8050
- SVC: Accuracy = 0.8150

**Best Model: SVC with Accuracy = 0.8150**

**Confusion matrices**
![Results](https://github.com/GaborWilk/BuyOrNot/blob/main/data/results_confusion_matrices.png?raw=true)

**Top features**
![Results](https://github.com/GaborWilk/BuyOrNot/blob/main/data/results_top_features.png?raw=true)

## Features

- Realistic dataset generation (age, income, gender)
- Full data preprocessing (numeric + categorical) to filter missing values
- Training multiple models (i.e. RandomForest, GradientBoosting, LogisticRegression and SVC) using pipelines
- Tunning hyperparameters with GridSearchCV cross-validation
- Evaluation with confusion matrix 
- Feature importance visualization
- Saving and reloading the model
- Making predictions on new data

## Tech Stack

- Python 3.10+
- pandas, numpy, matplotlib
- scikit-learn
- joblib

## Installation

```bash
git clone https://github.com/yourusername/BuyOrNot.git
cd BuyOrNot
pip install -r requirements.txt
```

## Usage:

1. Train the model
```python
python src/train_model.py
```

2. Make predictions
```python
python src/predict.py
```

### Example Prediction

```python
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

prediction = model.predict(new_person)

# Prediction Result
# ========================
# Will the person purchase? → Yes
# ========================
```

## License

MIT License. Feel free to fork and adapt.

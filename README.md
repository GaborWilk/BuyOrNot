# BuyOrNot

## Predicting Customer Purchase Behavior with Machine Learning Pipelines

**BuyOrNot** is a machine learning project that predicts whether a customer will make a purchase (it's generic, it could be a product, house, etc.) based on their demographics, using scikit-learn pipelines, preprocessing, and hyperparameter tuning. It is a binary classification problem, and basically the project gives an
answer to this question:
    Based on age, income, and gender, will this person make a purchase?

Where the output (purchased column) can be the following:
    1 = the person purchased (something)
    0 = the person did not purchase

It's designed for learning, experimentation, and potential real-world adaptation.

### Results
![Confusion Matrix](https://github.com/GaborWilk/BuyOrNot/blob/main/data/confusion_matrix.png?raw=true)
![Feature Importances](https://github.com/GaborWilk/BuyOrNot/blob/main/data/feature_importances.png?raw=true)

## Features

- Realistic dataset generation (age, income, gender)
- Full data preprocessing (numeric + categorical) to filter missing values
- Training a RandomForestClassifier using a pipeline
- Tunning hyperparameters with GridSearchCV cross-validation
- Evaluation with classification report + confusion matrix
- Feature importance visualization
- Saving and reloads the model
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
new_data = pd.DataFrame({
    'age': [35],
    'income': [75000],
    'gender': ['Female']
})

prediction = model.predict(new_data)
# Output: [1]  => Will Purchase
```

## License

MIT License. Feel free to fork and adapt.
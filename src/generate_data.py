import numpy as np
import pandas as pd


def generate_data():
    """
    input:
        'age': age of the person, e.g. 35
        'income': income of the person, e.g. 50000 Euro
        'gender': gender of the person, e.g. Male
        'education_level': education level of the person, e.g. High School
        'marital_status': marital status of the person, e.g. Single
        'children': the number of children the person has, e.g. 1
        'job_type': type of work the person has, e.g. Professional
        'credit_score': a formula using income/age based logic, e.g. 51000/30 = 1700
        'previous_purchase': number of previous purchases the person made, e.g. 2
        'interested_in_newsletter': whether the person is interested in newsletters, e.g. False

    output:
        'purchased': whether the person made the purchase, e.g. 1 (Yes)
    """

    # Generate realistic dataset
    np.random.seed(42)
    n_samples = 1000

    ages = np.random.randint(18, 70, size=n_samples)
    incomes = np.random.normal(loc=50000, scale=25000, size=n_samples).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], size=n_samples)
    education_levels = np.random.choice(
        ['Elementary', 'High School', 'Bachelor', 'Master', 'PhD'],
        size=n_samples
    )
    marital_statuses = np.random.choice(['Single', 'Married', 'Divorced', 'Widow'], size=n_samples)
    children = np.random.randint(0, 6, size=n_samples)
    job_types = np.random.choice(
        ['Student', 'Professional', 'Self-Employed', 'Unemployed'],
        size=n_samples
    )
    previous_purchases = np.random.randint(0, 100, size=n_samples)

    # improving realism by adding logic that rewards purchase history and penalizes dependents
    credit_scores = (incomes / ages) + (previous_purchases * 2) - (children * 5)
    credit_scores = np.clip(credit_scores, 300, 850)

    interested_in_newsletters = np.random.choice([True, False], size=n_samples)

    # Simulate purchase behavior
    purchase_prob = (
            0.25 * (incomes > 50000).astype(int) +
            0.20 * (ages.between(25, 45).astype(int)) +
            0.15 * (job_types == 'Student').astype(int) +
            0.16 * (job_types == 'Professional').astype(int) +
            0.18 * (job_types == 'Self-Employed').astype(int) +
            0.10 * (credit_scores > 575).astype(int) +
            0.10 * (previous_purchases > 2).astype(int) +
            0.10 * interested_in_newsletters.astype(int) +
            0.12 * (education_levels == 'Master').astype(int) +
            0.10 * (education_levels == 'PhD').astype(int)
    )

    # Add noise
    noise = np.random.normal(0, 0.1, size=n_samples)
    purchased = (purchase_prob + noise > 0.5).astype(int)

    data = pd.DataFrame({
        'age': ages,
        'income': incomes,
        'gender': genders,
        'education_level': education_levels,
        'marital_status': marital_statuses,
        'children': children,
        'job_type': job_types,
        'previous_purchase': previous_purchases,
        'credit_score': credit_scores,
        'interested_in_newsletter': interested_in_newsletters,
        'purchased': purchased
    })

    return data

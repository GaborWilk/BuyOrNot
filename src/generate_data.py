import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    incomes = np.random.normal(loc=50000, scale=20000, size=n_samples).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], size=n_samples)
    education_levels = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n_samples)
    marital_statuses = np.random.choice(['Single', 'Married', 'Divorced', 'Widow'], size=n_samples)
    children = np.random.randint(0, 5, size=n_samples)
    job_types = np.random.choice(['Student', 'Professional', 'Self-Employed', 'Unemployed'], size=n_samples)
    previous_purchases = np.random.randint(0, 50, size=n_samples)
    # improving realism by adding logic that rewards purchase history and penalizes dependents
    credit_scores = (incomes / ages) + (previous_purchases * 2) - (children * 4.35)
    credit_scores = np.clip(credit_scores, 300, 850)
    interested_in_newsletters = np.random.choice([True, False], size=n_samples)

    # Age related bonus
    age_score = np.zeros_like(ages, dtype=float)
    age_score[(ages >= 18) & (ages <= 35)] = 0.27
    age_score[(ages >= 36) & (ages <= 50)] = 0.15
    age_score[ages >= 51] = -0.05

    # Income related bonus
    income_score = (incomes - incomes.min()) / (incomes.max() - incomes.min())

    # Gender-related bonus
    gender_score = np.zeros_like(genders, dtype=float)
    gender_score[genders == 'Female'] = 0.15
    gender_score[genders == 'Male'] = 0.07

    # Education influence
    education_map = {'High School': 0.25, 'Bachelor': 0.3, 'Master': 0.5, 'PhD': 0.55}
    education_score = pd.Series(education_levels).map(education_map)

    # Job type influence
    job_map = {
        'Student': 0.35,
        'Professional': 0.45,
        'Self-Employed': 0.5,
        'Unemployed': -0.25
    }
    job_score = pd.Series(job_types).map(job_map)

    # Marital status and children influence
    family_score = ((marital_statuses == 'Married').astype(int) * 0.35) - (children * 0.07)
    family_score += ((marital_statuses == 'Single').astype(int) * 0.22) - (children * 0.09)

    # Credit and previous purchase influence
    credit_score_norm = (credit_scores - 300) / (850 - 300)
    purchase_history_score = (previous_purchases > 3).astype(int) * 0.085

    # Newsletter interest
    newsletter_score = interested_in_newsletters.astype(int) * 0.05

    # Combine all into a realistic probability
    purchase_prob = (
            0.20 * age_score +
            0.24 * income_score +
            0.11 * gender_score +
            0.16 * education_score +
            0.17 * job_score +
            0.18 * family_score +
            0.13 * credit_score_norm +
            0.11 * purchase_history_score +
            0.10 * newsletter_score
    )

    # Add noise and make binary
    purchase_prob = (purchase_prob - np.min(purchase_prob)) / (np.max(purchase_prob) - np.min(purchase_prob))
    purchase_prob = purchase_prob ** 1.4  # Skews more toward 0 and 1


    # Plot distribution of raw purchase probabilities
    plt.figure(figsize=(8, 5))
    plt.hist(purchase_prob, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Distribution of Purchase Probabilities (Before Noise)")
    plt.xlabel("Purchase Probability")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    #noise = np.random.normal(0, 0.225, size=n_samples)
    noise = np.random.normal(loc=0, scale=0.1, size=n_samples)
    purchased = (purchase_prob + noise > 0.5).astype(int)

    print(f"purchased bincount: {np.bincount(purchased)}")

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

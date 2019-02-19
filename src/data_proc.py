import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


def discretize_label_3(dosage):
    """
    Quantize the daily dosage to three discrete levels 0 ~ 2
    This function is consistent with both the project handout and the Lasso Bandit paper
    """
    if dosage <= 3:
        return 0  # low
    elif dosage >= 7:
        return 2  # high
    else:
        return 1  # medium

def discretize_label_9(dosage):
    """
    Quantize the daily dosage to nine discrete levels 0 ~ 8
    """
    dosage_round = round(dosage)
    if dosage_round <= 1:
        return 0
    elif dosage_round <= 8:
        return dosage_round - 1
    else:
        return 8

def discretize_label_dummy(dosage):
    return dosage


def preprocess(data, label_discretizer):
    """
    Extracts features and labels + imputes missing values for numerical features

    Args:
        data: DataFrame that contains raw Warfarin data
        label_discretizer: Function that maps continuous daily dosage to discrete levels (0 ~ K-1)
    Returns:
        features: features and labels extracted from the raw data

    Baseline Features:
    1. Demographic: Age, Height, Weight, Race
    2. Medication History: Enzyme, Amiodarone
    3. Genetic (used by Pharmacogenetic model): VKORC1, CYP2C9

    Additional Features (not used by the baseline linear models):
    1. Prior Warfarin treatment
    2. Smoking
    3. Congestive Heart Failure and/or Cardiomyopathy history

    Categorical variables are one-hot encoded, with "missing" being treated as a separate category

    Missing value percentage for numerical features: Age - 0.7%, Height - 19.6%, Weight - 4.9%
    The missing values for these features are mean-imputed
    """

    # Remove rows that don't have dosage labels
    data = data[~data['Therapeutic Dose of Warfarin'].isnull()]

    # Transform and quantize dosage labels
    features = data[['Therapeutic Dose of Warfarin']].copy()
    features['daily-dosage'] = features['Therapeutic Dose of Warfarin'] / 7
    features['dosage-level'] = features['daily-dosage'].map(label_discretizer)
    features.drop(['Therapeutic Dose of Warfarin'], axis=1, inplace=True)

    # Age
    features['Age'] = data['Age'].map(lambda x: float(x[:2]) / 10, na_action='ignore')
    features['Age'] = features['Age'].fillna(features['Age'].mean())

    # Height and Weight
    features['Height (cm)'] = data['Height (cm)'].fillna(data['Height (cm)'].mean())
    features['Weight (kg)'] = data['Weight (kg)'].fillna(data['Weight (kg)'].mean())

    # Race
    features['Asian'] = (data['Race'] == 'Asian').astype(int)
    features['African-American'] = (data['Race'] == 'Black or African American').astype(int)
    features['Race-Unknown'] = (data['Race'] == 'Unknown').astype(int)

    # Enzyme inducer status
    features['Enzyme'] = ((data['Carbamazepine (Tegretol)'] == 1) | (data['Phenytoin (Dilantin)'] == 1) |
                        (data['Rifampin or Rifampicin'] == 1)).astype(int)

    # Amiodarone status
    features['Amiodarone'] = (data['Amiodarone (Cordarone)'] == 1).astype(int)

    # Genetic features
    vkorc1 = data['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T']
    features['VKORC1-AG'] = (vkorc1 == 'A/G').astype(int)
    features['VKORC1-AA'] = (vkorc1 == 'A/A').astype(int)
    features['VKORC1-Unknown'] = (vkorc1.isnull()).astype(int)

    cyp2c9 = data['Cyp2C9 genotypes']
    features['CYP2C9-12'] = (cyp2c9 == '*1/*2').astype(int)
    features['CYP2C9-13'] = (cyp2c9 == '*1/*3').astype(int)
    features['CYP2C9-22'] = (cyp2c9 == '*2/*2').astype(int)
    features['CYP2C9-23'] = (cyp2c9 == '*2/*3').astype(int)
    features['CYP2C9-33'] = (cyp2c9 == '*3/*3').astype(int)
    features['CYP2C9-Unknown'] = (cyp2c9.isnull()).astype(int)

    # Additional features that are not used by the baseline models
    warfarin_treatment = data['Indication for Warfarin Treatment'].fillna('0')
    features['warfarin-treatment-3'] = warfarin_treatment.map(lambda x: 1 if '3' in x else 0, na_action='ignore')
    features['warfarin-treatment-4'] = warfarin_treatment.map(lambda x: 1 if '4' in x else 0, na_action='ignore')
    for col in ('Current Smoker', 'Congestive Heart Failure and/or Cardiomyopathy'):
        features[col + '-1'] = (data[col] == 1).astype(int)
        features[col + '-0'] = (data[col] == 0).astype(int)

    return features


def print_complete_percentage(df):
    """
    Prints (in order) how complete (i.e. no missing value) each feature is
    """
    complete_percent = []
    total_cnt = len(df)
    for col in df.columns:
        complete_cnt = total_cnt - (df[col].isnull()).sum()
        complete_percent.append((col, complete_cnt * 1.00 / total_cnt))
    complete_percent.sort(key=lambda x: x[1], reverse=True)
    for col, percent in complete_percent:
        print("{}: {}".format(col, percent))

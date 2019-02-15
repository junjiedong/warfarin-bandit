import numpy as np
import pandas as pd
from data_proc import discretize_label

def genetic_predict(data):
    """
    Make predictions using the baseline Pharmacogenetic dosing algorithm (Appx S1e)

    Args:
        data: DataFrame of features
    Returns:
        predicted_dosage_level: prediction output (Pandas Series)
    """
    predicted_dosage = (5.6044 -
                        0.2614 * data['Age'] +
                        0.0087 * data['Height (cm)'] +
                        0.0128 * data['Weight (kg)'] -
                        0.1092 * data['Asian'] -
                        0.2760 * data['African-American'] -
                        0.1032 * data['Race-Unknown'] +
                        1.1816 * data['Enzyme'] -
                        0.5503 * data['Amiodarone'] -
                        0.8677 * data['VKORC1-AG'] -
                        1.6974 * data['VKORC1-AA'] -
                        0.4854 * data['VKORC1-Unknown'] -
                        0.5211 * data['CYP2C9-12'] -
                        0.9357 * data['CYP2C9-13'] -
                        1.0616 * data['CYP2C9-22'] -
                        1.9206 * data['CYP2C9-23'] -
                        2.3312 * data['CYP2C9-33'] -
                        0.2188 * data['CYP2C9-Unknown'])**2 / 7

    predicted_dosage_level = predicted_dosage.map(discretize_label)
    return predicted_dosage_level


def clinical_predict(data):
    """
    Make predictions using the baseline Clinical dosing algorithm (Appx S1f)

    Args:
        data: DataFrame of features
    Returns:
        predicted_dosage_level: prediction output (Pandas Series)
    """
    predicted_dosage = (4.0376 -
                        0.2546 * data['Age'] +
                        0.0118 * data['Height (cm)'] +
                        0.0134 * data['Weight (kg)'] -
                        0.6752 * data['Asian'] +
                        0.4060 * data['African-American'] +
                        0.0443 * data['Race-Unknown'] +
                        1.2799 * data['Enzyme'] -
                        0.5695 * data['Amiodarone'])**2 / 7
    predicted_dosage_level = predicted_dosage.map(discretize_label)
    return predicted_dosage_level

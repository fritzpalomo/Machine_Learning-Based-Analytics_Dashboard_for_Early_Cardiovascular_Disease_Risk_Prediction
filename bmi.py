"""
bmi.py
------
BMI from height and weight, with the Asia-Pacific cut-offs the PhilPEN manual
uses (overweight from 23 kg/m2, obese from 25 kg/m2). Pure functions so they can be unit-tested.
"""


def compute_bmi(height_cm, weight_kg):
    """BMI rounded to 1 decimal, or None if either value is missing or not positive."""
    if not height_cm or not weight_kg or height_cm <= 0 or weight_kg <= 0:
        return None
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


def bmi_category(bmi):
    """Asia-Pacific category name for a BMI value, or None."""
    if bmi is None:
        return None
    if bmi < 18.5:
        return "Underweight"
    if bmi < 23:
        return "Normal"
    if bmi < 25:
        return "Overweight (at risk)"
    return "Obese"

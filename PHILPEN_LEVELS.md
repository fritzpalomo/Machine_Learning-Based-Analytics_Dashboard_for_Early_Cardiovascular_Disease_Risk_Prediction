# Five PhilPEN levels version

The dashboard now classifies into the five WHO/ISH (PhilPEN) chart levels, as the paper describes (Chapter IV 4.1.6, 4.1.8, 4.1.11), instead of Low / Moderate / High.

## What changed
- `prepare_enns_data.py`: builds `data/patients.csv` from the cleaned ENNS file (56,686 records, label = chart level). Already run; the CSV is included.
- `train_model.py`: household-grouped 80/20 split (seed 42); only XGBoost is tuned (five settings, three-fold household-grouped CV); the other four models use fixed settings; XGBoost is the deployed family; SHAP explainer saved as before.
- `preprocessing.py`: ten predictors (age, sex, SBP, DBP, total cholesterol, glucose, BMI, smoking, alcohol, physical activity). Family history is not a model input (not in ENNS); the form still records it with the case.
- `levels.py`, `philpen_chart.py`, `who_ish_wprb_digitised.json`: level names, colours, Composite Risk Score (weights 0/25/50/75/100) and the WHO/ISH chart reference.
- `app.py`: Level 1 to Level 5 result box, five-band gauge, chart reference with the effect of one systolic band higher or lower, PhilPEN management panel, five Reports cards. Alcohol options are now Never/Former/Current and physical activity High/Low, the values in ENNS.
- Survey changes (height/weight BMI, medication, case code, plain-language help, diet/glucose cards) are kept.

## To run (needs xgboost and shap, which were not installable where this was built)
    pip install -r requirements.txt
    python train_model.py        # writes models/*.joblib
    streamlit run app.py
Delete any old `models/` and the old `predictions` database rows if you want a clean Reports page (old rows with Low/Moderate/High still display).

## Not verified here
The Streamlit pages and the XGBoost/SHAP training could not be run in the build environment. The 77 logic tests passed with a stand-in model.

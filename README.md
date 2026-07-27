# Multiple Disease Prediction System Using Machine Learning
<img width="1366" height="700" alt="image" src="https://github.com/user-attachments/assets/1ad44cfa-d817-437c-936e-dfeb18080764" />

## Project Title
Multiple Disease Prediction System Using Machine Learning

## Group Members
- Saad Jamil
- Ahmed Siddiqui
- Wania Imran

## Objective
To build a single web application that predicts a patient's likelihood of having any of six diseases — diabetes, heart disease, Parkinson's disease, breast cancer, chronic kidney disease, and liver disease — from structured clinical measurements. Each disease is handled by an independently trained classical machine learning model (Support Vector Machine or Logistic Regression), and all six are served through one interactive Streamlit interface.

## Required Libraries
Listed in `requirements.txt`:
- `streamlit` — web application framework
- `streamlit-option-menu==0.3.2` — sidebar navigation menu component
- `scikit-learn==1.0.2` — model training and inference (pinned to match the version models were trained/pickled with)
- `pandas` — data loading and cleaning
- `numpy` — numeric array handling

## Installation Steps

1. Clone or download this repository, and open a terminal in the project folder.

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the following files are present in the same folder as `multiplediseaseprediction.py` (all included in this submission):
   - `diabetes_model.sav`
   - `heart_disease_model.sav`
   - `parkinsons_model.sav`
   - `breast_cancer_model.sav`
   - `kidney_disease_model.sav`
   - `liver_disease_model.sav`
   - `liver_disease_scaler.sav`

## How to Run the Project

1. From the project folder, run:
   ```bash
   streamlit run multiplediseaseprediction.py
   ```

2. Streamlit will start a local server and print a URL (typically `http://localhost:8501`) — open it in a web browser if it doesn't open automatically.

3. Use the sidebar navigation menu to select a disease to test.

4. Fill in the requested numeric clinical values (placeholder examples are shown in each field).

5. Click the **Predict** button for that disease to see the result.

### Re-training a model (optional)
Each disease has its own training notebook (`Diabetes_Model.ipynb`, `Heart_Model.ipynb`, `Parkinson_Model.ipynb`, `Breast_Cancer_Model.ipynb`, `Kidney_Model.ipynb`, `Liver_Model_FIXED.ipynb`). Running a notebook end-to-end regenerates its corresponding `.sav` file(s) from the raw dataset CSV.

## Expected Output

- A browser tab opens showing **"Multiple Disease Prediction System"** with a sidebar listing all six diseases.
- Selecting a disease shows a form of labelled numeric input fields matching that model's training features.
- After entering values and clicking Predict, the app displays either:
  - A **green success box** — e.g. "The person is not diabetic", or
  - A **red error box** — e.g. "The person is diabetic"
- If a required model file is missing, the app shows a clear error message for that specific disease page rather than crashing the whole application.
- Approximate test-set accuracy by disease (see project report for full details): Diabetes ~77%, Heart Disease ~82%, Parkinson's ~87%, Breast Cancer ~97%, Kidney Disease ~93%, Liver Disease evaluated primarily via confusion matrix/F1-score due to class imbalance.

## Project Structure
```
├── multiplediseaseprediction.py     # Main Streamlit application
├── requirements.txt                  # Python dependencies
├── Diabetes_Model.ipynb              # Training notebook — Diabetes
├── Heart_Model.ipynb                 # Training notebook — Heart Disease
├── Parkinson_Model.ipynb             # Training notebook — Parkinson's
├── Breast_Cancer_Model.ipynb         # Training notebook — Breast Cancer
├── Kidney_Model.ipynb                # Training notebook — Kidney Disease
├── Liver_Model_FIXED.ipynb           # Training notebook — Liver Disease
├── diabetes.csv / heart.csv / parkinsons.csv
│   / breast_cancer.csv / kidney_disease.csv / liver_disease.csv   # Datasets
├── *_model.sav                       # Trained models (pickled)
└── liver_disease_scaler.sav          # Fitted StandardScaler for the liver model
```

## Acknowledgement
This project is done with six disease modules built by this team: diabetes,heart disease,parkinson disease ,breast cancer, chronic kidney disease, and liver disease, each with its own dataset, data-cleaning pipeline, and trained model, integrated into a single shared application.

## Disclaimer
This is an educational course project, not a validated medical diagnostic tool. Predictions should not be used for real clinical decision-making.

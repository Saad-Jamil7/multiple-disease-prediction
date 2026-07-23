import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Configure the Streamlit page layout first
st.set_page_config(page_title="Multiple Disease Prediction System", layout="wide")

# Custom CSS injector targeting modern Streamlit containers (.stApp)
def set_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), url('{url}') no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background safely using standard container targeting
set_bg_from_url("https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768")

# Loading the saved models safely
models = {}

def safely_load_model(model_name, filename):
    try:
        return pickle.load(open(filename, 'rb'))
    except Exception as e:
        return None

models['diabetes'] = safely_load_model('diabetes', 'diabetes_model.sav')
models['heart'] = safely_load_model('heart', 'heart_disease_model.sav')
models['parkinsons'] = safely_load_model('parkinsons', 'parkinsons_model.sav')
models['breast'] = safely_load_model('breast', 'breast_cancer_model.sav')
models['kidney'] = safely_load_model('kidney', 'kidney_disease_model.sav')
models['liver'] = safely_load_model('liver', 'liver_disease_model.sav')

def safely_load_scaler(filename):
    try:
        return pickle.load(open(filename, 'rb'))
    except Exception:
        return None

liver_scaler = safely_load_scaler('liver_disease_scaler.sav')
# ----------------------------------

# Sidebar navigation design
with st.sidebar:
    selected = option_menu(
        'Navigation Menu',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction',
         'Breast Cancer Prediction',
         'Kidney Disease Prediction',
         'Liver Disease Prediction'],
        icons=['activity', 'heart', 'person', 'gender-female', 'droplet', 'clipboard2-pulse'],
        default_index=0
    )

st.title("Multiple Disease Prediction System")
st.markdown("---")

# 1. Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.subheader('Diabetes Prediction')
    col1, col2, col3 = st.columns(3)
    
    with col1: Pregnancies = st.text_input('Number of Pregnancies', placeholder='e.g. 2')
    with col2: Glucose = st.text_input('Glucose Level', placeholder='e.g. 120')
    with col3: BloodPressure = st.text_input('Blood Pressure value', placeholder='e.g. 70')
    with col1: SkinThickness = st.text_input('Skin Thickness value', placeholder='e.g. 20')
    with col2: Insulin = st.text_input('Insulin Level', placeholder='e.g. 79')
    with col3: BMI = st.text_input('BMI value', placeholder='e.g. 25.5')
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', placeholder='e.g. 0.47')
    with col2: Age = st.text_input('Age of the Person', placeholder='e.g. 33')
    
    # FIXED: Indented this block so it belongs strictly inside the Diabetes page condition
    if st.button('Diabetes Test Result'):
        if models['diabetes'] is None:
            st.error("This model file is incompatible or missing. Please re-train it inside its notebook first!")
        else:
            try:
                inputs = [float(x) if x else 0.0 for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
                diab_prediction = models['diabetes'].predict([inputs])
                if diab_prediction[0] == 1:
                    st.error('The person is diabetic')
                else:
                    st.success('The person is not diabetic')
            except Exception as e:
                st.warning("Please ensure all inputs are numeric.")

# 2. Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.subheader('Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)
    
    with col1: age = st.text_input('Age', placeholder='e.g. 54')
    with col2: sex = st.text_input('Sex (1 = Male, 0 = Female)', placeholder='e.g. 1')
    with col3: cp = st.text_input('Chest Pain types (0-3)', placeholder='e.g. 0')
    with col1: trestbps = st.text_input('Resting Blood Pressure', placeholder='e.g. 130')
    with col2: chol = st.text_input('Serum Cholestoral in mg/dl', placeholder='e.g. 246')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', placeholder='e.g. 0')
    with col1: restecg = st.text_input('Resting Electrocardiographic results (0-2)', placeholder='e.g. 1')
    with col2: thalach = st.text_input('Maximum Heart Rate achieved', placeholder='e.g. 150')
    with col3: exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)', placeholder='e.g. 0')
    with col1: oldpeak = st.text_input('ST depression induced by exercise', placeholder='e.g. 1.0')
    with col2: slope = st.text_input('Slope of the peak exercise ST segment (0-2)', placeholder='e.g. 1')
    with col3: ca = st.text_input('Major vessels colored by flourosopy (0-4)', placeholder='e.g. 0')
    with col1: thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', placeholder='e.g. 2')
    
    if st.button('Heart Disease Test Result'):
        if models['heart'] is None:
            st.error("This model file is incompatible or missing. Please re-train it inside its notebook first!")
        else:
            try:
                inputs = [float(x) if x else 0.0 for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
                heart_prediction = models['heart'].predict([inputs])
                if heart_prediction[0] == 0:
                    st.error('The person has heart disease')
                else:
                    st.success('The person does not have heart disease')
            except Exception as e:
                st.warning("Please ensure all inputs are numeric.")

# 3. Parkinson's Prediction Page
elif selected == "Parkinsons Prediction":
    st.subheader("Parkinson's Disease Prediction")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1: fo = st.text_input('MDVP:Fo(Hz)', placeholder='e.g. 119.99')
    with col2: fhi = st.text_input('MDVP:Fhi(Hz)', placeholder='e.g. 157.30')
    with col3: flo = st.text_input('MDVP:Flo(Hz)', placeholder='e.g. 74.99')
    with col4: Jitter_percent = st.text_input('MDVP:Jitter(%)', placeholder='e.g. 0.0078')
    with col5: Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', placeholder='e.g. 0.00007')
    with col1: RAP = st.text_input('MDVP:RAP', placeholder='e.g. 0.0037')
    with col2: PPQ = st.text_input('MDVP:PPQ', placeholder='e.g. 0.0055')
    with col3: DDP = st.text_input('Jitter:DDP', placeholder='e.g. 0.0111')
    with col4: Shimmer = st.text_input('MDVP:Shimmer', placeholder='e.g. 0.0437')
    with col5: Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', placeholder='e.g. 0.426')
    with col1: APQ3 = st.text_input('Shimmer:APQ3', placeholder='e.g. 0.0219')
    with col2: APQ5 = st.text_input('Shimmer:APQ5', placeholder='e.g. 0.0248')
    with col3: APQ = st.text_input('MDVP:APQ', placeholder='e.g. 0.0357')
    with col4: DDA = st.text_input('Shimmer:DDA', placeholder='e.g. 0.0657')
    with col5: NHR = st.text_input('NHR', placeholder='e.g. 0.0208')
    with col1: HNR = st.text_input('HNR', placeholder='e.g. 21.03')
    with col2: RPDE = st.text_input('RPDE', placeholder='e.g. 0.4147')
    with col3: DFA = st.text_input('DFA', placeholder='e.g. 0.8153')
    with col4: spread1 = st.text_input('spread1', placeholder='e.g. -4.8130')
    with col5: spread2 = st.text_input('spread2', placeholder='e.g. 0.2664')
    with col1: D2 = st.text_input('D2', placeholder='e.g. 2.3014')
    with col2: PPE = st.text_input('PPE', placeholder='e.g. 0.2846')
    
    if st.button("Parkinson's Test Result"):
        if models['parkinsons'] is None:
            st.error("This model file is incompatible or missing. Please re-train it inside its notebook first!")
        else:
            try:
                inputs = [float(x) if x else 0.0 for x in [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
                parkinsons_prediction = models['parkinsons'].predict([inputs])
                if parkinsons_prediction[0] == 1:
                    st.error("The person has Parkinson's disease")
                else:
                    st.success("The person does not have Parkinson's disease")
            except Exception as e:
                st.warning("Please ensure all inputs are numeric.")

# 4. Breast Cancer Prediction Page
elif selected == 'Breast Cancer Prediction':
    st.subheader('Breast Cancer Prediction')
    st.caption('Enter the cell nuclei measurements computed from a digitized image.')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1: mean_radius = st.text_input('Mean Radius', placeholder='e.g. 14.1')
    with col2: mean_texture = st.text_input('Mean Texture', placeholder='e.g. 19.3')
    with col3: mean_perimeter = st.text_input('Mean Perimeter', placeholder='e.g. 91.9')
    with col4: mean_area = st.text_input('Mean Area', placeholder='e.g. 654.9')
    with col5: mean_smoothness = st.text_input('Mean Smoothness', placeholder='e.g. 0.096')
    with col1: mean_compactness = st.text_input('Mean Compactness', placeholder='e.g. 0.104')
    with col2: mean_concavity = st.text_input('Mean Concavity', placeholder='e.g. 0.089')
    with col3: mean_concave_points = st.text_input('Mean Concave Points', placeholder='e.g. 0.048')
    with col4: mean_symmetry = st.text_input('Mean Symmetry', placeholder='e.g. 0.181')
    with col5: mean_fractal_dimension = st.text_input('Mean Fractal Dimension', placeholder='e.g. 0.063')
    with col1: radius_error = st.text_input('Radius Error', placeholder='e.g. 0.405')
    with col2: texture_error = st.text_input('Texture Error', placeholder='e.g. 1.216')
    with col3: perimeter_error = st.text_input('Perimeter Error', placeholder='e.g. 2.866')
    with col4: area_error = st.text_input('Area Error', placeholder='e.g. 40.34')
    with col5: smoothness_error = st.text_input('Smoothness Error', placeholder='e.g. 0.007')
    with col1: compactness_error = st.text_input('Compactness Error', placeholder='e.g. 0.025')
    with col2: concavity_error = st.text_input('Concavity Error', placeholder='e.g. 0.032')
    with col3: concave_points_error = st.text_input('Concave Points Error', placeholder='e.g. 0.012')
    with col4: symmetry_error = st.text_input('Symmetry Error', placeholder='e.g. 0.020')
    with col5: fractal_dimension_error = st.text_input('Fractal Dimension Error', placeholder='e.g. 0.004')
    with col1: worst_radius = st.text_input('Worst Radius', placeholder='e.g. 16.3')
    with col2: worst_texture = st.text_input('Worst Texture', placeholder='e.g. 25.4')
    with col3: worst_perimeter = st.text_input('Worst Perimeter', placeholder='e.g. 107.3')
    with col4: worst_area = st.text_input('Worst Area', placeholder='e.g. 880.6')
    with col5: worst_smoothness = st.text_input('Worst Smoothness', placeholder='e.g. 0.132')
    with col1: worst_compactness = st.text_input('Worst Compactness', placeholder='e.g. 0.254')
    with col2: worst_concavity = st.text_input('Worst Concavity', placeholder='e.g. 0.273')
    with col3: worst_concave_points = st.text_input('Worst Concave Points', placeholder='e.g. 0.114')
    with col4: worst_symmetry = st.text_input('Worst Symmetry', placeholder='e.g. 0.290')
    with col5: worst_fractal_dimension = st.text_input('Worst Fractal Dimension', placeholder='e.g. 0.084')
    
    if st.button('Breast Cancer Test Result'):
        if models['breast'] is None:
            st.error("This model file is incompatible or missing. Please re-train it inside its notebook first!")
        else:
            try:
                inputs = [float(x) if x else 0.0 for x in [
                    mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity,
                    mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error,
                    smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                    worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity,
                    worst_concave_points, worst_symmetry, worst_fractal_dimension
                ]]
                breast_cancer_prediction = models['breast'].predict([inputs])
                if breast_cancer_prediction[0] == 0:
                    st.error('The mass is malignant (cancerous)')
                else:
                    st.success('The mass is benign (non-cancerous)')
            except Exception as e:
                st.warning("Please ensure all inputs are numeric.")

# 5. Kidney Disease Prediction Page
elif selected == 'Kidney Disease Prediction':
    st.subheader('Kidney Disease Prediction')
    col1, col2, col3 = st.columns(3)
    
    with col1: age_k = st.text_input('Age', placeholder='e.g. 48', key='age_k')
    with col2: bp = st.text_input('Blood Pressure (mm/Hg)', placeholder='e.g. 80')
    with col3: sg = st.text_input('Specific Gravity', placeholder='e.g. 1.02')
    with col1: al = st.text_input('Albumin (0-5)', placeholder='e.g. 1')
    with col2: su = st.text_input('Sugar (0-5)', placeholder='e.g. 0')
    with col3: bgr = st.text_input('Blood Glucose Random (mgs/dl)', placeholder='e.g. 121')
    with col1: bu = st.text_input('Blood Urea (mgs/dl)', placeholder='e.g. 36')
    with col2: sc = st.text_input('Serum Creatinine (mgs/dl)', placeholder='e.g. 1.2')
    with col3: sod = st.text_input('Sodium (mEq/L)', placeholder='e.g. 137')
    with col1: pot = st.text_input('Potassium (mEq/L)', placeholder='e.g. 4.6')
    with col2: hemo = st.text_input('Hemoglobin (gms)', placeholder='e.g. 15.4')
    with col3: pcv = st.text_input('Packed Cell Volume', placeholder='e.g. 44')
    with col1: wc = st.text_input('White Blood Cell Count (cells/cumm)', placeholder='e.g. 7800')
    with col2: rc = st.text_input('Red Blood Cell Count (millions/cmm)', placeholder='e.g. 5.2')
    
    if st.button('Kidney Disease Test Result'):
        if models['kidney'] is None:
            st.error("This model file is incompatible or missing. Please re-train it inside its notebook first!")
        else:
            try:
                inputs = [float(x) if x else 0.0 for x in [age_k, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc]]
                kidney_prediction = models['kidney'].predict([inputs])
                if kidney_prediction[0] == 1:
                    st.error('The person has chronic kidney disease')
                else:
                    st.success('The person does not have chronic kidney disease')
            except Exception as e:
                st.warning("Please ensure all inputs are numeric.")

# 6. Liver Disease Prediction Page
elif selected == 'Liver Disease Prediction':
    st.subheader('Liver Disease Prediction')
    col1, col2, col3 = st.columns(3)
    
    with col1: age_l = st.text_input('Age', placeholder='e.g. 45', key='age_l')
    with col2: gender_l = st.text_input('Gender (1 = Male, 0 = Female)', placeholder='e.g. 1')
    with col3: total_bilirubin = st.text_input('Total Bilirubin', placeholder='e.g. 0.7')
    with col1: direct_bilirubin = st.text_input('Direct Bilirubin', placeholder='e.g. 0.2')
    with col2: alkaline_phosphotase = st.text_input('Alkaline Phosphotase', placeholder='e.g. 187')
    with col3: alamine_aminotransferase = st.text_input('Alamine Aminotransferase', placeholder='e.g. 16')
    with col1: aspartate_aminotransferase = st.text_input('Aspartate Aminotransferase', placeholder='e.g. 18')
    with col2: total_protiens = st.text_input('Total Proteins', placeholder='e.g. 6.8')
    with col3: albumin = st.text_input('Albumin', placeholder='e.g. 3.3')
    with col1: albumin_and_globulin_ratio = st.text_input('Albumin and Globulin Ratio', placeholder='e.g. 0.9')
    
if st.button('Liver Disease Test Result'):
        if models['liver'] is None or liver_scaler is None:
            st.error("Model or scaler file is missing. Please re-run the training notebook first!")
        else:
            try:
                # 1. Collect inputs normally
                inputs = [float(x) if x else 0.0 for x in [
                    age_l, gender_l, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                    alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                    albumin, albumin_and_globulin_ratio
                ]]
                
                # 2. Scale the inputs before feeding them to the model
                inputs_scaled = liver_scaler.transform([inputs])
                
                # 3. Predict using the scaled inputs
                liver_prediction = models['liver'].predict(inputs_scaled)
                
                if liver_prediction[0] == 1:
                    st.error('The person has liver disease')
                else:
                    st.success('The person does not have liver disease')
            except Exception as e:
                st.warning("Please ensure all inputs are numeric.")
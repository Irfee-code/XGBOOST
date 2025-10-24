import streamlit as st
import pickle
import pandas as pd
import os

# --- 1. LOAD THE SAVED MODEL ---

MODEL_FILE = 'xgb.pkl' 

@st.cache_resource
def load_model(model_path):
    """
    Loads the pickled model.
    st.cache_resource ensures this only runs once.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}")
        st.stop()
        
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- 2. DEFINE THE FEATURE LIST (FROM YOUR ERROR) ---
# This is the exact list of 44 features your model was trained on.
MODEL_FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently',
    'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
    'Department_Sales', 'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
    'Gender_Male', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'OverTime_Yes'
]

# --- 3. CREATE THE INPUT WIDGETS ---
def user_input_features():
    """
    Creates sidebar widgets for all features.
    Returns a dictionary of the raw user inputs.
    """
    st.sidebar.header("Employee Details")
    
    # --- Numerical Features ---
    # We take all numerical features from the MODEL_FEATURES list
    st.sidebar.subheader("Numerical Features")
    Age = st.sidebar.slider("Age", 18, 60, 35)
    DailyRate = st.sidebar.slider("Daily Rate", 100, 1500, 800)
    DistanceFromHome = st.sidebar.slider("Distance From Home (miles)", 1, 30, 10)
    HourlyRate = st.sidebar.slider("Hourly Rate", 30, 100, 65)
    MonthlyIncome = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
    MonthlyRate = st.sidebar.slider("Monthly Rate", 2000, 27000, 14000)
    NumCompaniesWorked = st.sidebar.slider("Number of Companies Worked", 0, 9, 2)
    PercentSalaryHike = st.sidebar.slider("Percent Salary Hike", 11, 25, 15)
    TotalWorkingYears = st.sidebar.slider("Total Working Years", 0, 40, 10)
    TrainingTimesLastYear = st.sidebar.slider("Training Times Last Year", 0, 6, 3)
    YearsAtCompany = st.sidebar.slider("Years At Company", 0, 40, 5)
    YearsInCurrentRole = st.sidebar.slider("Years In Current Role", 0, 18, 4)
    YearsSinceLastPromotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
    YearsWithCurrManager = st.sidebar.slider("Years With Current Manager", 0, 17, 4)

    # --- Ordinal Features (as sliders) ---
    st.sidebar.subheader("Rating Features (1-5)")
    Education = st.sidebar.slider("Education Level", 1, 5, 3)
    EnvironmentSatisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
    JobInvolvement = st.sidebar.slider("Job Involvement", 1, 4, 3)
    JobLevel = st.sidebar.slider("Job Level", 1, 5, 2)
    JobSatisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
    PerformanceRating = st.sidebar.slider("Performance Rating", 1, 4, 3)
    RelationshipSatisfaction = st.sidebar.slider("Relationship Satisfaction", 1, 4, 3)
    StockOptionLevel = st.sidebar.slider("Stock Option Level", 0, 3, 1)
    WorkLifeBalance = st.sidebar.slider("Work Life Balance", 1, 4, 3)

    # --- Categorical Features (as dropdowns) ---
    # We MUST infer the original categories from the one-hot encoded feature names
    st.sidebar.subheader("Categorical Features")
    
    # Note: The 'drop_first=True' base case (e.g., 'Non-Travel') is just omitted
    BusinessTravel = st.sidebar.selectbox(
        "Business Travel",
        ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel') 
    )
    Department = st.sidebar.selectbox(
        "Department",
        ('Sales', 'Research & Development', 'Human Resources') 
    )
    EducationField = st.sidebar.selectbox(
        "Education Field",
        ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources')
    )
    Gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    JobRole = st.sidebar.selectbox(
        "Job Role",
        ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 
         'Manufacturing Director', 'Research Director', 'Manager', 
         'Sales Representative', 'Human Resources', 'Healthcare Representative') # Guessed 9th
    )
    MaritalStatus = st.sidebar.selectbox(
        "Marital Status",
        ('Married', 'Single', 'Divorced') 
    )
    OverTime = st.sidebar.selectbox("Over Time", ('Yes', 'No'))

    # Store all inputs in a dictionary
    raw_data = {
        'Age': Age, 'DailyRate': DailyRate, 'DistanceFromHome': DistanceFromHome, 
        'Education': Education, 'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'HourlyRate': HourlyRate, 'JobInvolvement': JobInvolvement, 'JobLevel': JobLevel,
        'JobSatisfaction': JobSatisfaction, 'MonthlyIncome': MonthlyIncome,
        'MonthlyRate': MonthlyRate, 'NumCompaniesWorked': NumCompaniesWorked,
        'PercentSalaryHike': PercentSalaryHike, 'PerformanceRating': PerformanceRating,
        'RelationshipSatisfaction': RelationshipSatisfaction, 'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': TotalWorkingYears, 'TrainingTimesLastYear': TrainingTimesLastYear,
        'WorkLifeBalance': WorkLifeBalance, 'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole, 'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        
        # Categorical data
        'BusinessTravel': BusinessTravel,
        'Department': Department,
        'EducationField': EducationField,
        'Gender': Gender,
        'JobRole': JobRole,
        'MaritalStatus': MaritalStatus,
        'OverTime': OverTime
    }
    return raw_data


def preprocess_input(raw_data):
    """
    Converts the raw data dictionary into a one-hot encoded
    DataFrame that matches the model's 44 expected features.
    """
    # 1. Initialize a dictionary with all 44 features set to 0
    processed_data = {feature: 0 for feature in MODEL_FEATURES}

    # 2. Populate the numerical/ordinal values directly
    for key in raw_data:
        if key in processed_data:
            processed_data[key] = raw_data[key]

    # 3. Manually set the one-hot encoded flags based on user selection
    # This assumes 'drop_first=True' was used during training, so we
    # only set the flag for the non-base categories.

    # BusinessTravel
    if raw_data['BusinessTravel'] == 'Travel_Frequently':
        processed_data['BusinessTravel_Travel_Frequently'] = 1
    elif raw_data['BusinessTravel'] == 'Travel_Rarely':
        processed_data['BusinessTravel_Travel_Rarely'] = 1
    # 'Non-Travel' is the base case (both are 0)

    # Department
    if raw_data['Department'] == 'Research & Development':
        processed_data['Department_Research & Development'] = 1
    elif raw_data['Department'] == 'Sales':
        processed_data['Department_Sales'] = 1
    # 'Human Resources' is the base case

    # EducationField
    if raw_data['EducationField'] == 'Life Sciences':
        processed_data['EducationField_Life Sciences'] = 1
    elif raw_data['EducationField'] == 'Marketing':
        processed_data['EducationField_Marketing'] = 1
    elif raw_data['EducationField'] == 'Medical':
        processed_data['EducationField_Medical'] = 1
    elif raw_data['EducationField'] == 'Other':
        processed_data['EducationField_Other'] = 1
    elif raw_data['EducationField'] == 'Technical Degree':
        processed_data['EducationField_Technical Degree'] = 1
    # 'Human Resources' is the base case
    
    # Gender
    if raw_data['Gender'] == 'Male':
        processed_data['Gender_Male'] = 1
    # 'Female' is the base case

    # JobRole (This is complex, must match all 8)
    if raw_data['JobRole'] == 'Human Resources':
        processed_data['JobRole_Human Resources'] = 1
    elif raw_data['JobRole'] == 'Laboratory Technician':
        processed_data['JobRole_Laboratory Technician'] = 1
    elif raw_data['JobRole'] == 'Manager':
        processed_data['JobRole_Manager'] = 1
    elif raw_data['JobRole'] == 'Manufacturing Director':
        processed_data['JobRole_Manufacturing Director'] = 1
    elif raw_data['JobRole'] == 'Research Director':
        processed_data['JobRole_Research Director'] = 1
    elif raw_data['JobRole'] == 'Research Scientist':
        processed_data['JobRole_Research Scientist'] = 1
    elif raw_data['JobRole'] == 'Sales Executive':
        processed_data['JobRole_Sales Executive'] = 1
    elif raw_data['JobRole'] == 'Sales Representative':
        processed_data['JobRole_Sales Representative'] = 1
    # 9th role 'Healthcare Representative' is the base case
        
    # MaritalStatus
    if raw_data['MaritalStatus'] == 'Married':
        processed_data['MaritalStatus_Married'] = 1
    elif raw_data['MaritalStatus'] == 'Single':
        processed_data['MaritalStatus_Single'] = 1
    # 'Divorced' is the base case

    # OverTime
    if raw_data['OverTime'] == 'Yes':
        processed_data['OverTime_Yes'] = 1
    # 'No' is the base case

    # 4. Convert the dictionary to a 1-row DataFrame
    # The columns will be in the exact order as MODEL_FEATURES
    input_df = pd.DataFrame([processed_data], columns=MODEL_FEATURES)
    return input_df


# --- 4. SET UP THE APP UI ---

st.set_page_config(
    page_title="Employee Attrition Prediction App",
    page_icon="🤖",
    layout="wide"
)

st.title("Employee Attrition Prediction App 🚀")

# Load the model
model = load_model(MODEL_FILE)
st.success(f"Model '{MODEL_FILE}' loaded successfully!")

# --- 5. CREATE TABS FOR DIFFERENT MODES ---

tab1, tab2 = st.tabs(["Single Prediction (Manual Input)", "Batch Prediction (File Upload)"])

# --- TAB 1: MANUAL INPUT FOR A SINGLE PREDICTION ---
with tab1:
    st.header("Predict a Single Employee's Attrition")
    
    # Get inputs from sidebar
    raw_input_data = user_input_features()
    
    # Display the raw inputs
    st.subheader("Current Input Values (Raw)")
    st.json(raw_input_data)
    
    # Prediction button
    if st.button("Predict Single Case", type="primary", use_container_width=True):
        
        # Pre-process the raw data
        try:
            input_df = preprocess_input(raw_input_data)
            
            st.subheader("Processed Model Input (Top 5 features)")
            st.dataframe(input_df.iloc[:, :5]) # Show first 5 processed features

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] # Prob of class 1

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"**Prediction: ATTRITION (Class 1)**")
                st.write("This employee is at a high risk of leaving.")
            else:
                st.success(f"**Prediction: NO ATTRITION (Class 0)**")
                st.write("This employee is likely to stay.")
            
            st.metric(label="Probability of Attrition (Class 1)", value=f"{probability:.2%}")
            
        except Exception as e:
            st.error(f"An error occurred during pre-processing or prediction:")
            st.exception(e)

# --- TAB 2: YOUR ORIGINAL FILE UPLOADER ---
with tab2:
    st.header("Predict from a CSV File (Batch Prediction)")
    st.write("Upload a CSV file with **raw data** (e.g., 'Sales', 'Male').")
    st.write("This app will automatically pre-process it.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV into a DataFrame
            input_df_raw = pd.read_csv(uploaded_file)
            
            st.subheader("Input Data (from CSV)")
            st.dataframe(input_df_raw.head(), use_container_width=True)
            
            st.header("Run Prediction on CSV")
            
            if st.button("Predict on CSV", type="primary", use_container_width=True):
                with st.spinner("Pre-processing and making predictions..."):
                    
                    # --- Pre-process the entire DataFrame ---
                    # We can't use the same `preprocess_input` function,
                    # so we use pd.get_dummies and reindex
                    
                    # 1. Apply get_dummies to the raw input
                    input_df_dummies = pd.get_dummies(input_df_raw, drop_first=True)
                    
                    # 2. Align columns with the model's features
                    # This adds missing columns (as 0s) and drops extra ones
                    input_df_processed = input_df_dummies.reindex(
                        columns=MODEL_FEATURES, 
                        fill_value=0
                    )
                    
                    # Use the loaded model to make predictions
                    predictions = model.predict(input_df_processed)
                    probabilities = model.predict_proba(input_df_processed)[:, 1]
                    
                    st.header("Prediction Results")
                    
                    # Create a results DataFrame
                    results_df = input_df_raw.copy()
                    results_df['Prediction'] = predictions
                    results_df['Probability_Attrition'] = probabilities
                    
                    # Display the results
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Provide a download button for the results
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name='prediction_results.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please check your file format and column names.")

    else:
        st.info("Awaiting CSV file upload...")


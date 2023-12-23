import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Fetch and Prepare Data
@st.cache_data
def load_data():
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    return X, y

X, y = load_data()

# Define Preprocessing Steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Initialize and Train Model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42, max_depth=20, max_features='sqrt', n_estimators=300))])
model.fit(X, y)

# Streamlit App
def main():
    st.title("Bank Marketing Prediction App")
    
    # User Inputs for Prediction
    age = st.number_input("Age", min_value=18, max_value=95, value=30)
    job = st.selectbox("Job Type", options=['admin.', 'blue-collar', 'entrepreneur', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'housemaid', 'unknown'])
    marital_status = st.selectbox("Marital Status", options=["married", "single", "divorced"])
    education = st.selectbox("Education", options=["primary", "secondary", "tertiary", "unknown"])
    default = st.radio("Has Credit in Default?", options=["yes", "no"])
    balance = st.number_input("Average Yearly Balance", min_value=-10000, max_value=100000, value=0)
    housing = st.radio("Has Housing Loan?", options=["yes", "no"])
    loan = st.radio("Has Personal Loan?", options=["yes", "no"])
    contact = st.selectbox("Contact Communication Type", options=["cellular", "telephone", "unknown"])
    day_of_week = st.number_input("Last Contact Day of the Week", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last Contact Month", options=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    duration = st.number_input("Last Contact Duration (in seconds)", min_value=0, max_value=5000, value=100)
    campaign = st.number_input("Number of Contacts During This Campaign", min_value=1, max_value=100, value=1)
    pdays = st.number_input("Number of Days Passed After Being Last Contacted", min_value=-1, max_value=1000, value=0)
    previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, max_value=300, value=0)
    poutcome = st.selectbox("Outcome of Previous Marketing Campaign", options=["failure", "other", "success", "unknown"])

    # When the user inputs all required fields, you can make a prediction:
    if st.button('Predict'):
        # Create a DataFrame or Series from user inputs
        input_data = pd.DataFrame([[age, job, marital_status, education, default, balance, housing, loan, contact, day_of_week, month, duration, campaign, pdays, previous, poutcome]], 
                                  columns=numeric_features.tolist() + categorical_features.tolist())
        
        # Make Prediction
        prediction = model.predict(input_data)
        st.write(f"The predicted outcome is: {'Yes' if prediction[0] == 'yes' else 'No'}")
        
        if __name__ == "__main__":
            main()
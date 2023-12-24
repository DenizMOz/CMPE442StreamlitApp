import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
# Fetch and Prepare Data
@st.cache_data
def load_data():
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    X=X.drop(columns=['duration'])
    X['no_previous_contact'] = X['pdays'].apply(lambda x: 1 if x == 999 else 0)
    X=X.drop(columns=['pdays'])

    return X, y
# Define preprocessing steps and train model
@st.cache_data()
def initialize_and_train_model(X, y):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), 
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)), 
    ('scaler', MinMaxScaler()),
    ('feature_selector', SelectKBest(score_func=f_classif, k=10))
    ])

    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42, max_depth=20, max_features='sqrt', n_estimators=300))])
    model.fit(X, y)
    return model
# Load data and train model
X, y = load_data()
model = initialize_and_train_model(X, y)



def main():
    st.title("Bank Marketing Prediction App")
    st.header("Predictive Analysis")
    st.text("Use the inputs below to predict marketing outcomes.")

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
    day_of_week = st.number_input("Last Contact Day of the Month", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last Contact Month", options=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    contacted = st.radio("Contacted Previously?", options=["yes", "no"])
    contacted_numeric = 1 if contacted == 'yes' else 0
    campaign = st.number_input("Number of Contacts During This Campaign", min_value=1, max_value=100, value=1)
    previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, max_value=300, value=0)
    poutcome = st.selectbox("Outcome of Previous Marketing Campaign", options=["failure", "other", "success", "unknown"])
    
    # When the user inputs all required fields, you can make a prediction:
    if st.button('Predict'):
        # Create a DataFrame or Series from user inputs
        input_data = pd.DataFrame([[
            age, balance, contacted_numeric ,campaign, previous,  # Numeric features
            job, marital_status, education, default, housing, loan, contact, day_of_week, month, poutcome  # Categorical features
        ]], columns=[
            'age', 'balance', 'no_previous_contact', 'campaign', 'previous',  # Numeric feature names
            'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'month', 'poutcome'  # Categorical feature names
        ])
        
        # Make Prediction
        prediction = model.predict(input_data)
        st.write(f"The predicted outcome is: {'Yes' if prediction[0] == 'yes' else 'No'}")

if __name__ == "__main__":
    main()

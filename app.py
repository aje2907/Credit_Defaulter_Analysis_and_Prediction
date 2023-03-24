# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import joblib

# Load the trained model
model = joblib.load("my_model.joblib")

# Define encoding and scaling functions
def encode_categorical_features(df):
    categorical_columns = ['NAME_FAMILY_STATUS', 'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'PREV_CLUST']
    df_cat = df[['NAME_FAMILY_STATUS', 'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'PREV_CLUST']]
    df_num = df.drop(['NAME_FAMILY_STATUS', 'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'PREV_CLUST'], axis = 1)

    df_family_status = pd.get_dummies(df_cat['NAME_FAMILY_STATUS'])
    df_gender = pd.get_dummies(df_cat['CODE_GENDER'])
    df_education = pd.get_dummies(df_cat['NAME_EDUCATION_TYPE'])
    df_income = pd.get_dummies(df_cat['NAME_INCOME_TYPE'])
    df_prev_cluster = pd.get_dummies(df_cat['PREV_CLUST'])
    df_final_categorical_encoded = pd.concat([df_family_status, df_gender, df_education, df_income, df_prev_cluster], axis=1)
    df = pd.concat([df_final_categorical_encoded, df_num], axis = 1)
    return df

def scale_numerical_features(df):
    numerical_columns = ['DAYS_BIRTH_y', 'DAYS_EMPLOYED_y', 'DAYS_ID_PUBLISH_y', 'DAYS_LAST_PHONE_CHANGE_y']
    df_num = df[['DAYS_BIRTH_y', 'DAYS_EMPLOYED_y', 'DAYS_ID_PUBLISH_y', 'DAYS_LAST_PHONE_CHANGE_y']]
    df_cat = df.drop(['DAYS_BIRTH_y', 'DAYS_EMPLOYED_y', 'DAYS_ID_PUBLISH_y', 'DAYS_LAST_PHONE_CHANGE_y'], axis = 1)

    scaler = joblib.load('scaler_weights.joblib')
    df_final_numerical_scaled = scaler.transform(df_num)
    df_final_numerical_scaled = pd.DataFrame(df_final_numerical_scaled,columns=numerical_columns)
    df = pd.concat([df_cat, df_final_numerical_scaled], axis = 1)
    return df

# Create Flask app
app = Flask(__name__)

# Define the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Collect input from user
    input_dict = {
        'NAME_FAMILY_STATUS': request.form['NAME_FAMILY_STATUS'],
        'CODE_GENDER': request.form['CODE_GENDER'],
        'NAME_EDUCATION_TYPE': request.form['NAME_EDUCATION_TYPE'],
        'NAME_INCOME_TYPE': request.form['NAME_INCOME_TYPE'],
        'PREV_CLUST': request.form['PREV_CLUST'],
        'DAYS_BIRTH_y': float(request.form['DAYS_BIRTH_y']),
        'DAYS_EMPLOYED_y': float(request.form['DAYS_EMPLOYED_y']),
        'DAYS_ID_PUBLISH_y': float(request.form['DAYS_ID_PUBLISH_y']),
        'DAYS_LAST_PHONE_CHANGE_y': float(request.form['DAYS_LAST_PHONE_CHANGE_y'])
    }

    # Convert input dictionary to a pandas DataFrame
    input_df = pd.DataFrame.from_dict(input_dict, orient='index').T
    
    # Encode categorical features
    input_df = encode_categorical_features(input_df)
    
    # Scale numerical features
    input_df = scale_numerical_features(input_df)

    # Modifying df to accomodate fit for customer cluster
    if input_dict['PREV_CLUST'] == 'a':
        input_df['b'] = 0
        input_df['c'] = 0
        input_df['d'] = 0
    elif input_dict['PREV_CLUST'] == 'b':
        input_df['a'] = 0
        input_df['c'] = 0
        input_df['d'] = 0
    elif input_dict['PREV_CLUST'] == 'c':
        input_df['a'] = 0
        input_df['b'] = 0
        input_df['d'] = 0
    elif input_dict['PREV_CLUST'] == 'd':
        input_df['a'] = 0
        input_df['b'] = 0
        input_df['c'] = 0

    # Modifying df to accomodate fit for NAME_FAMILY_STATUS ['Single / not married', 'Married', 'Civil marriage', 'Widow','Separated']
    if input_dict['NAME_FAMILY_STATUS'] == "Civil marriage":
        input_df['Single / not married'] = 0
        input_df['Married'] = 0
        input_df['Widow'] = 0
        input_df['Separated'] = 0
    elif input_dict['NAME_FAMILY_STATUS'] == "Married":
        input_df['Single / not married'] = 0
        input_df['Civil marriage'] = 0
        input_df['Widow'] = 0
        input_df['Separated'] = 0
    elif input_dict['NAME_FAMILY_STATUS'] == "Separated":
        input_df['Single / not married'] = 0
        input_df['Civil marriage'] = 0
        input_df['Married'] = 0
        input_df['Widow'] = 0
    elif input_dict['NAME_FAMILY_STATUS'] == "Single / not married":
        input_df['Widow'] = 0
        input_df['Civil marriage'] = 0
        input_df['Married'] = 0
        input_df['Separated'] = 0
    elif input_dict['NAME_FAMILY_STATUS'] == "Widow":
        input_df['Single / not married'] = 0
        input_df['Civil marriage'] = 0
        input_df['Married'] = 0
        input_df['Separated'] = 0
    # Modifying df to accomodate fit for CODE_GENDER
    if input_dict['CODE_GENDER'] == "F":
        input_df['M'] = 0
        input_df['XNA'] = 0
    elif input_dict['CODE_GENDER'] == "M":
        input_df['F'] = 0
        input_df['XNA'] = 0
    elif input_dict['CODE_GENDER'] == "XNA":
        input_df['F'] = 0
        input_df['M'] = 0
    # Modifying df to accomodate fit for NAME_EDUCATION_TYPE ['Secondary / secondary special', 'Higher education','Incomplete higher', 'Lower secondary', 'Academic degree']
    if input_dict['NAME_EDUCATION_TYPE'] == "Academic degree":
        input_df['Secondary / secondary special'] = 0
        input_df['Higher education'] = 0
        input_df['Incomplete higher'] = 0
        input_df['Lower secondary'] = 0
    elif input_dict['NAME_EDUCATION_TYPE'] == "Higher education":
        input_df['Secondary / secondary special'] = 0
        input_df['Academic degree'] = 0
        input_df['Incomplete higher'] = 0
        input_df['Lower secondary'] = 0
    elif input_dict['NAME_EDUCATION_TYPE'] == "Incomplete higher":
        input_df['Secondary / secondary special'] = 0
        input_df['Academic degree'] = 0
        input_df['Higher education'] = 0
        input_df['Lower secondary'] = 0
    elif input_dict['NAME_EDUCATION_TYPE'] == "Lower secondary":
        input_df['Secondary / secondary special'] = 0
        input_df['Academic degree'] = 0
        input_df['Higher education'] = 0
        input_df['Incomplete higher'] = 0
    elif input_dict['NAME_EDUCATION_TYPE'] == "Secondary / secondary special":
        input_df['Lower secondary'] = 0
        input_df['Academic degree'] = 0
        input_df['Higher education'] = 0
        input_df['Incomplete higher'] = 0

    # Modifying df to accomodate fit for NAME_INCOME_TYPE ['Working', 'State servant', 'Commercial associate', 'Student','Pensioner', 'Businessman', 'Maternity leave']
    if input_dict['NAME_INCOME_TYPE'] == "Businessman":
        input_df['Commercial associate'] = 0
        input_df['Maternity leave'] = 0
        input_df['Pensioner'] = 0
        input_df['State servant'] = 0
        input_df['Student'] = 0
        input_df['Working'] = 0
    elif input_dict['NAME_INCOME_TYPE'] == "Commercial associate":
        input_df['Businessman'] = 0
        input_df['Maternity leave'] = 0
        input_df['Pensioner'] = 0
        input_df['State servant'] = 0
        input_df['Student'] = 0
        input_df['Working'] = 0 
    elif input_dict['NAME_INCOME_TYPE'] == "Maternity leave":
        input_df['Businessman'] = 0
        input_df['Commercial associate'] = 0
        input_df['Pensioner'] = 0
        input_df['State servant'] = 0
        input_df['Student'] = 0
        input_df['Working'] = 0 
    elif input_dict['NAME_INCOME_TYPE'] == "Pensioner":
        input_df['Businessman'] = 0
        input_df['Commercial associate'] = 0
        input_df['Maternity leave'] = 0
        input_df['State servant'] = 0
        input_df['Student'] = 0
        input_df['Working'] = 0 
    elif input_dict['NAME_INCOME_TYPE'] == "State servant":
        input_df['Businessman'] = 0
        input_df['Commercial associate'] = 0
        input_df['Maternity leave'] = 0
        input_df['Pensioner'] = 0
        input_df['Student'] = 0
        input_df['Working'] = 0 
    elif input_dict['NAME_INCOME_TYPE'] == "Student":
        input_df['Businessman'] = 0
        input_df['Commercial associate'] = 0
        input_df['Maternity leave'] = 0
        input_df['Pensioner'] = 0
        input_df['State servant'] = 0
        input_df['Working'] = 0 
    elif input_dict['NAME_INCOME_TYPE'] == "Working":
        input_df['Businessman'] = 0
        input_df['Commercial associate'] = 0
        input_df['Maternity leave'] = 0
        input_df['Pensioner'] = 0
        input_df['State servant'] = 0
        input_df['Student'] = 0

    # reordering df to the order of fit

    input_df = input_df[['Civil marriage',
                            'Married',
                            'Separated',
                            'Single / not married',
                            'Widow',
                            'F',
                            'M',
                            'XNA',
                            'Academic degree',
                            'Higher education',
                            'Incomplete higher',
                            'Lower secondary',
                            'Secondary / secondary special',
                            'Businessman',
                            'Commercial associate',
                            'Maternity leave',
                            'Pensioner',
                            'State servant',
                            'Student',
                            'Working',
                            'a',
                            'b',
                            'c',
                            'd',
                            'DAYS_BIRTH_y',
                            'DAYS_EMPLOYED_y',
                            'DAYS_ID_PUBLISH_y',
                            'DAYS_LAST_PHONE_CHANGE_y']]
    # Make predictions
    prediction = model.predict(input_df)
    
    # Convert prediction to a human-readable format
    if prediction[0] == 0:
        result = "Loan approved"
    else:
        result = "Loan not approved"
        
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

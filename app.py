import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
try:
    with open('ridge_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None # Set model to None if loading fails

# Define the list of selected features used during training
# This list should match the columns in X_train_selected
selected_features = ['Annual Income', 'Health Score', 'Credit Score', 'Age', 'Policy_Start_Day', 'Vehicle Age', 'Policy_Start_Month', 'Insurance Duration', 'Policy_Start_Year', 'Number of Dependents', 'Previous Claims', 'Gender_male', 'Smoking Status_yes', 'Location_suburban', 'Marital Status_married', 'Policy Type_premium', 'Occupation_unemployed', 'Customer Feedback_good', 'Marital Status_single', 'Education Level_phd']


# Streamlit application layout
st.title("Insurance Premium Prediction")
st.header("Enter the details to predict the insurance premium amount")

# Create input fields for the selected features
user_inputs = {}

# Categorical features that were one-hot encoded (need to handle these specifically)
categorical_inputs_info = {
    'Gender': ['male'], # Assuming 'female' is the base/dropped category
    'Smoking Status': ['yes'], # Assuming 'no' is the base
    'Location': ['suburban', 'urban'], # Assuming 'rural' is the base
    'Marital Status': ['married', 'single'], # Assuming 'divorced' is the base
    'Policy Type': ['premium'], # Assuming 'basic' and 'comprehensive' are bases (based on feature list)
    'Occupation': ['unemployed'], # Assuming 'employed' and 'self-employed' are bases
    'Customer Feedback': ['good', 'poor'], # Assuming 'average' is the base
    'Education Level': ["high school", "master's", "phd"], # Assuming 'bachelor's' is the base
    # 'Property Type': ['condo', 'house'] # Assuming 'apartment' is the base - not in selected features
}

# Collect inputs for numerical features
st.subheader("Numerical Features")
for feature in selected_features:
    if feature not in [item for sublist in categorical_inputs_info.values() for item in sublist] and \
       feature not in categorical_inputs_info.keys() and \
       not any(feature.startswith(cat_col.replace(' ', '_').lower() + '_') for cat_col in categorical_inputs_info.keys()):
        # Determine appropriate default value or range based on feature
        # For simplicity, using a generic number input; in a real app, use domain knowledge
        user_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Collect inputs for categorical features with one-hot encoding
st.subheader("Categorical Features")
for original_col, categories in categorical_inputs_info.items():
    # Check if any of the one-hot encoded columns for this original column are in selected_features
    if any(f"{original_col.replace(' ', '_').lower()}_{cat}" in selected_features for cat in categories):
        selected_category = st.selectbox(f"Select {original_col}", ['Select'] + categories + [cat for cat in df_encoded[original_col.replace(' ', '_').lower()].unique() if cat not in categories and cat != 'select']) # Add other categories if they exist in the original data but not in the selected features' one-hot encoded list

        # Initialize one-hot encoded columns for this original column to 0 (False)
        for cat in categories:
            encoded_col_name = f"{original_col.replace(' ', '_').lower()}_{cat}"
            if encoded_col_name in selected_features:
                 user_inputs[encoded_col_name] = False # Initialize as boolean False

        # Set the selected category's one-hot encoded column to 1 (True)
        if selected_category != 'Select':
             encoded_col_name = f"{original_col.replace(' ', '_').lower()}_{selected_category}"
             if encoded_col_name in selected_features:
                 user_inputs[encoded_col_name] = True # Set as boolean True
             elif selected_category in [cat for cat in df_encoded[original_col.replace(' ', '_').lower()].unique() if cat not in categories and cat != 'select']:
                  # Handle the case where a selected category is not among the one-hot encoded features in selected_features
                  # This might happen if the selected features didn't include all categories of an original column
                  # For simplicity here, we'll just print a warning; in a real app, you might need to rethink feature selection or encoding
                  st.warning(f"Selected category '{selected_category}' for '{original_col}' is not one of the selected features for the model.")


# Create a DataFrame from user inputs, ensuring correct order and dtypes
if model is not None:
    input_df = pd.DataFrame([user_inputs])

    # Ensure the columns in the input DataFrame match the order and presence of selected_features
    # Add any missing selected features with a default value (e.g., 0 or False)
    for feature in selected_features:
        if feature not in input_df.columns:
            # Determine if the missing feature is likely numerical or boolean (from one-hot encoding)
            # This is a heuristic; a more robust approach would store the dtypes of X_train_selected
            if any(feature.startswith(cat_col.replace(' ', '_').lower() + '_') for cat_col in categorical_inputs_info.keys()):
                 input_df[feature] = False # Assume missing one-hot encoded feature is False
            else:
                 input_df[feature] = 0.0 # Assume missing numerical feature is 0.0

    # Reorder columns to match the training data
    input_df = input_df[selected_features]

    # Convert boolean columns to appropriate type if necessary for the model
    # Linear models often work with float/int, tree models can handle bool
    # Let's convert boolean columns to int (0 or 1) for broader compatibility
    for col in input_df.columns:
        if input_df[col].dtype == 'bool':
            input_df[col] = input_df[col].astype(int)


    # Prediction button
    if st.button("Predict Premium Amount"):
        try:
            prediction = model.predict(input_df)
            st.success(f"Predicted Premium Amount: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Instructions on how to run
st.markdown("""
### How to run the Streamlit app:

1. Save the code above as `app.py`.
2. Make sure you have Streamlit and the necessary libraries installed (`pip install streamlit pandas numpy scikit-learn`).
3. Ensure the saved model file (`ridge_best_model.pkl`) is in the same directory as `app.py`.
4. Open your terminal or command prompt, navigate to the directory containing the files, and run:
   ```bash
   streamlit run app.py
   ```
""")

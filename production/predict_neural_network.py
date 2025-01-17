import joblib
import pandas as pd

# Load the neural network pipeline
pipeline_path = 'models/nn_pipeline.pkl'
try:
    nn_pipeline = joblib.load(pipeline_path)
    print(f"Pipeline loaded successfully from {pipeline_path}.")
except FileNotFoundError:
    raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}. Ensure the model is trained and saved.")

def predict_single_nn(raw_data: dict):
    """
    Predicts the outcome for a single input row of raw data using the neural network.
    Args:
        raw_data (dict): A dictionary representing a single row of input data.
    Returns:
        dict: The predicted class and probabilities.
    """
    # Convert raw data to a DataFrame
    input_df = pd.DataFrame([raw_data])
    
    # Make predictions
    predicted_class = nn_pipeline.predict(input_df)[0]
    predicted_proba = nn_pipeline.predict_proba(input_df)[0]
    
    return {
        "predicted_class": predicted_class,
        "probabilities": {
            "class_0": predicted_proba[0],
            "class_1": predicted_proba[1]
        }
    }



# Define a function to preprocess and predict
def predict_multiple_nn(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict using the trained neural network pipeline.
    
    Parameters:
        input_data (pd.DataFrame): Raw input data.
        
    Returns:
        pd.DataFrame: Predictions with class probabilities.
    """
    # Ensure input_data is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")

    # Make predictions
    probabilities = nn_pipeline.predict_proba(input_data)
    predictions = nn_pipeline.predict(input_data)

    # Create output DataFrame
    output = input_data.copy()
    output['Predicted Class'] = predictions
    output['Probability Yes'] = probabilities[:, 1]
    output['Probability No'] = probabilities[:, 0]

    return output




# Example usage
if __name__ == "__main__":
    # Example input data
    data_line = {
        "age": 25,
        "sex": "Male",
        "race": "Caucasian",
        "juv_fel_count": 5,
        "juv_misd_count": 0,
        "juv_other_count": 0,
        "priors_count": 7,
        "c_charge_degree": "F",
        "age_cat": "25 - 45",
        "c_jail_in": "2020-01-01 00:00:00",
        "c_jail_out": "2020-01-02 00:00:00",
        "in_custody": "2020-01-01 00:00:00",
        "out_custody": "2020-01-02 00:00:00"
    }

    # Predict using the neural network pipeline
    result = predict_single_nn(data_line)
    print("Prediction Result:", result)



    test_data = pd.DataFrame([
        {
            "age": 25,
            "sex": "Male",
            "race": "Caucasian",
            "juv_fel_count": 5,
            "juv_misd_count": 0,
            "juv_other_count": 0,
            "priors_count": 7,
            "c_charge_degree": "F",
            "age_cat": "25 - 45",
            "c_jail_in": "2020-01-01 00:00:00",
            "c_jail_out": "2020-01-02 00:00:00",
            "in_custody": "2020-01-01 00:00:00",
            "out_custody": "2020-01-02 00:00:00"
        },
        {
            "age": 19,
            "sex": "Female",
            "race": "African-American",
            "juv_fel_count": 2,
            "juv_misd_count": 1,
            "juv_other_count": 1,
            "priors_count": 3,
            "c_charge_degree": "M",
            "age_cat": "Less than 25",
            "c_jail_in": "2020-02-15 00:00:00",
            "c_jail_out": "2020-02-17 00:00:00",
            "in_custody": "2020-02-14 00:00:00",
            "out_custody": "2020-02-18 00:00:00"
        },
        {
            "age": 45,
            "sex": "Male",
            "race": "Hispanic",
            "juv_fel_count": 0,
            "juv_misd_count": 0,
            "juv_other_count": 0,
            "priors_count": 1,
            "c_charge_degree": "F",
            "age_cat": "Greater than 45",
            "c_jail_in": "2021-05-01 00:00:00",
            "c_jail_out": "2021-05-10 00:00:00",
            "in_custody": "2021-04-30 00:00:00",
            "out_custody": "2021-05-11 00:00:00"
        }
    ])
    
    result = predict_multiple_nn(test_data)
    print("Prediction Result:", result)
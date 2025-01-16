import joblib
import pandas as pd

# Load the pipeline
pipeline_path = 'models/logistic_pipeline.pkl'
logistic_pipeline = joblib.load(pipeline_path)

def predict_single(raw_data: dict):
    """
    Predicts the outcome for a single input row of raw data.
    
    Parameters:
        raw_data (dict): A dictionary representing a single row of input data.
                        Example:
                        {
                            "age": 25,
                            "sex": "Male",
                            "race": "Caucasian",
                            "juv_fel_count": 0,
                            "juv_misd_count": 0,
                            "juv_other_count": 0,
                            "priors_count": 2,
                            "c_charge_degree": "F",
                            "age_cat": "25 - 45"
                        }

    Returns:
        dict: The predicted class and probabilities.
    """
    # Convert raw data to a DataFrame
    input_df = pd.DataFrame([raw_data])
    
    # Make predictions
    predicted_class = logistic_pipeline.predict(input_df)[0]
    predicted_proba = logistic_pipeline.predict_proba(input_df)[0]
    
    return {
        "predicted_class": predicted_class,
        "probabilities": {
            "class_0": predicted_proba[0],
            "class_1": predicted_proba[1]
        }
    }

# Example usage
if __name__ == "__main__":
    data_line = {
        "age": 25,
        "sex": "Male",
        "race": "Caucasian",
        "juv_fel_count": 5,
        "juv_misd_count": 0,
        "juv_other_count": 0,
        "priors_count": 7,
        "c_charge_degree": "F",
        "age_cat": "25 - 45"
    }

    result = predict_single(data_line)
    print("Prediction Result:", result)

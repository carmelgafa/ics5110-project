import gradio as gr
import joblib
import numpy as np

# Load your pipeline
# pipeline = joblib.load('model_pipeline.pkl')

import predict_knn
import predict_neural_network
import predict_logistic_regression

INPUT_FIELDS = [
    "age",
    "sex",
    "race",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
    "age_cat",
    "c_jail_in",
    "c_jail_out",
    "in_custody",
    "out_custody",
]


# Define a function to make predictions
def predict(*features):
    
    inputs = {}
    
    for i, field in enumerate(INPUT_FIELDS):
        inputs[field] = features[i]
        
    print(predict_neural_network.predict_single(inputs)["predicted_class"])
        
    outputs = [
    "Likely to reoffend" if predict_knn.predict_single(inputs)["predicted_class"] == 1 else "Unlikely to reoffend",
    "Likely to reoffend" if predict_neural_network.predict_single(inputs)["predicted_class"] == 1 else "Unlikely to reoffend",
    "Likely to reoffend" if predict_logistic_regression.predict_single(inputs)["predicted_class"] == 1 else "Unlikely to reoffend",
    ]
    
    return outputs

inputs = [
    gr.Textbox(label="age", value=25),
    gr.Dropdown(label="sex", value="M", choices=["M", "F"]),
    gr.Dropdown(label="race", value="Caucasian", choices=["Caucasian", "African-American", "Hispanic", "Other"]),
    gr.Textbox(label="juv_fel_count", value=0),
    gr.Textbox(label="juv_misd_count", value=0),
    gr.Textbox(label="juv_other_count", value=0),
    gr.Textbox(label="priors_count", value=0),
    gr.Dropdown(label="c_charge_degree", value="F", choices=["F", "M"]),
    gr.Dropdown(label="age_cat", value="25 - 45", choices=["Less than 25", "25 - 45", "Greater than 45"]),
    gr.Textbox(label="c_jail_in", value="2020-01-01 00:00:00"),
    gr.Textbox(label="c_jail_out", value="2020-01-02 00:00:00"),
    gr.Textbox(label="in_custody", value="2020-01-01 00:00:00"),
    gr.Textbox(label="out_custody", value="2020-01-02 00:00:00"),
    
]


# Define the output component
outputs = [
    gr.Textbox(label="kNN prediction"),
    gr.Textbox(label="Neural network prediction"),
    gr.Textbox(label="Logistic regression prediction")
]

# Create the Gradio interface
app = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Pipeline Predictor")

if __name__ == "__main__":
    app.launch(share=True)

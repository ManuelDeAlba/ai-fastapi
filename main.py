from fastapi import FastAPI
import Orange
import joblib
import numpy as np

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    model = joblib.load("iris-tree-classification.pkcls")

    input_data = np.array([data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]])
    prediction = model(input_data)

    return {"prediction": model.domain.class_var.values[int(prediction)]}
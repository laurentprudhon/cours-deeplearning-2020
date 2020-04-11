from fastapi import FastAPI, File, UploadFile
from pathlib import Path
from bears import BearClassifier

app = FastAPI()

model_path = Path()/"bears.pkl"
bc = BearClassifier(model_path)

@app.post("/bear/")
def predict_bear(file: UploadFile = File(...)):
    return bc.predict(file)

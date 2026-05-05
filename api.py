from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from main import analyze_text,analyze_nutrition
import json


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# load model
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# input structure
class FoodInput(BaseModel):
    ingredients: str
    energy: float
    sugar: float
    fat: float
    protein: float
    salt: float
    fiber: float
    carbs: float

@app.post("/analyze")
def analyze_food(data: FoodInput):
    nlp_score, reasons = analyze_text(data.ingredients)

    ml_result = analyze_nutrition(
        data.energy,
        data.sugar,
        data.fat,
        data.protein,
        data.salt,
        data.fiber,
        data.carbs
    )

    return {
        "ml_result": ml_result,
        "nlp_score": nlp_score,
        "reasons": reasons
    }
# NLP


@app.post("/predict")
def predict(data: FoodInput):

    score = analyze_text(data.ingredients)

    features = [[
        data.energy,
        data.sugar,
        data.fat,
        data.protein,
        data.salt,
        data.fiber,
        data.carbs
    ]]

    prediction = model.predict(features)[0]
    ml_result = label_encoder.inverse_transform([prediction])[0]

    if ml_result == "Unhealthy" or score < -2:
        final = "Unhealthy"
    elif ml_result == "Healthy" and score > 0:
        final = "Healthy"
    else:
        final = "Moderate"
    save_history(data.ingredients, final, score)

    return {
        "result": final,
        "ml": ml_result,
        "score": score
    }
def save_history(text, result, score):
    data = {
        "ingredients": text,
        "result": result,
        "score": score
    }

    try:
        with open("history.json", "r") as f:
            history = json.load(f)
    except:
        history = []

    history.append(data)

    with open("history.json", "w") as f:
        json.dump(history, f, indent=4)

@app.get("/history")
def get_history():
    try:
        with open("history.json", "r") as f:
            return json.load(f)
    except:
        return []
import joblib
import pandas as pd
import re

# Load trained model + encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ---------------- NLP PART ----------------
def analyze_text(text):
    text = text.lower()

    score = 0
    reasons = []

    # ✅ first check combined words
    if "palm oil" in text:
        score -= 3
        reasons.append("⚠️ palm oil")

        # remove it so it doesn't count again
        text = text.replace("palm oil", "")

    tokens = text.split()

    unhealthy = ["sugar", "syrup", "glucose", "fructose", "oil", "palm", "fat"]
    healthy = ["oats", "nuts", "fruit", "fiber", "protein"]

    for word in tokens:
        if word in unhealthy:
            score -= 2
            reasons.append(f"⚠️ {word}")
        elif word in healthy:
            score += 1
            reasons.append(f"✅ {word}")

    return score, reasons

# ---------------- ML PART ----------------
def analyze_nutrition():
    print("\nEnter nutrition values (per 100g):")

    energy = float(input("Energy: "))
    sugar = float(input("Sugar: "))
    fat = float(input("Fat: "))
    protein = float(input("Protein: "))
    salt = float(input("Salt: "))
    fiber = float(input("Fiber: "))
    carbs = float(input("Carbs: "))

    data = pd.DataFrame([[energy, sugar, fat, protein, salt, fiber, carbs]],
                        columns=[
                            "energy_100g",
                            "sugars_100g",
                            "fat_100g",
                            "proteins_100g",
                            "salt_100g",
                            "fiber_100g",
                            "carbohydrates_100g"
                        ])

    pred = model.predict(data)[0]
    label = label_encoder.inverse_transform([pred])[0]

    return label

# ---------------- MAIN ----------------
print("🍔 Food Health Analyzer\n")

text = input("Enter ingredients: ")

nlp_score, reasons = analyze_text(text)
ml_result = analyze_nutrition()

print("\n--- FINAL RESULT ---")

if ml_result == "Unhealthy" and nlp_score < -2:
    print("❌ Unhealthy Food")

elif ml_result == "Healthy" and nlp_score > 0:
    print("✅ Healthy Food")

else:
    print("⚖️ Moderate Food")

print("\nReasons:")
seen=set()

for r in reasons:
    if r not in seen:
        print(r)
        seen.add(r)

print(f"\nScore: {nlp_score}")
if nlp_score <=-3:
    print("⚠️ High Risk Ingredients Detected")
elif nlp_score >=2:
    print("💪 Healthy ingredient profile")
print(f"ML Prediction: {ml_result}")
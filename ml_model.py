import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
import re

df = pd.read_csv("en.openfoodfacts.org.products.tsv", sep="\t", nrows=5000)
df = df[["ingredients_text", "nutrition_grade_fr"]].dropna()

# label create
def convert_label(grade):
    if grade in ["a", "b"]:
        return 1
    elif grade in ["d", "e"]:
        return 0
    else:
        return -1  # moderate

df["label"] = df["nutrition_grade_fr"].apply(convert_label)

# remove moderate for now (optional)
df = df[df["label"] != -1]

# load dataset
df=df[df["ingredients_text"].str.len()>10]
texts = df["ingredients_text"]
labels = df["label"]   # 1 = healthy, 0 = unhealthy

# vectorize
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

# model
model = MultinomialNB()
model.fit(X, labels)

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens
def get_reasons(tokens, text):
    unhealthy = ["sugar", "syrup", "glucose", "fructose", "maltodextrin",
    "refined", "oil", "palm", "fat", "fried",
    "preservative", "emulsifier", "stabilizer",
    "flavour", "color", "additive",
    "sodium", "salt"]
    healthy = ["oats", "nuts", "fruit", "fiber", "protein"]
    score=0
    reasons = []

    for word in tokens:
        if word in unhealthy:
            score-=2
            reasons.append(f"⚠️ {word}")
        elif word in healthy:
            score+=1
            reasons.append(f"✅ {word}")
    if "palm oil" in text:
        score-=3
        reasons.append("⚠️ palm oil ")
    if "refined flour" in text:
        reasons.append("⚠️ refined flour ")
    if "whole oats"in text:
        score+=2
        reasons.append("✅ whole oats ")

    # INS detection
    ins = re.findall(r'(e\d+|ins\s?\d+)', text)
    for i in ins:
        reasons.append(f"⚠️ additive ({i})")

    return score,reasons



def predict_health(text):
    X_test = vectorizer.transform([text])
    pred = model.predict(X_test)[0]
    prob=model.predict_proba(X_test)[0]
    confidence=max(prob)
    return pred,confidence


def show_result(pred, reasons):
    print("\n--- Analysis Result ---")

    if pred == 1:
        print("✅ Healthy Food")
    else:
        print("❌ Unhealthy Food")

    print("\nReasons:")
    for r in set(reasons):
        print(r)
text = input("Enter ingredients: ")

prediction, confidence = predict_health(text)

tokens = preprocess(text)
score, reasons = get_reasons(tokens, text)

print("\n--- Analysis Result ---")

if prediction == 0 or score < -2:
    print("❌ Unhealthy Food")
elif prediction == 1 or score > 2:
    print("✅ Healthy Food")
else:
    print("⚖️ Moderate")

print(f"Confidence: {round(confidence*100,2)}%")
print(f"Score: {score}")

print("\nReasons:")
for r in set(reasons):
    print(r)


import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re

text = input("Enter ingredients: ").lower()

tokens = word_tokenize(text)

# 🔥 create bigrams (2-word phrases)
bigrams = [" ".join(pair) for pair in ngrams(tokens, 2)]

# 🚫 unhealthy
unhealthy_keywords = [
    "sugar", "syrup", "glucose", "fructose", "maltodextrin",
    "refined oil", "palm oil", "fried food",
    "preservative", "emulsifier", "stabilizer",
    "flavour", "color", "additive"
]

# ✅ healthy
healthy_keywords = [
    "fiber", "protein", "vitamin", "iron",
    "whole grain", "whole oats", "brown rice",
    "fruit", "vegetable", "nuts", "seeds"
]

score = 0
reasons = []

# 🔍 check phrases first
for phrase in bigrams:
    if phrase in unhealthy_keywords:
        score -= 3
        reasons.append(f"⚠️ {phrase}")
    elif phrase in healthy_keywords:
        score += 2
        reasons.append(f"✅ {phrase}")

# 🔍 check single words
for word in tokens:
    if word in unhealthy_keywords:
        score -= 2
        reasons.append(f"⚠️ {word}")
    elif word in healthy_keywords:
        score += 1
        reasons.append(f"✅ {word}")

# 🔍 INS detection
ins_matches = re.findall(r'(e\d+|ins\s?\d+)', text)
for item in ins_matches:
    score -= 2
    reasons.append(f"⚠️ additive ({item})")

# 🎯 result
print("\n--- Analysis Result ---")

if score < -3:
    print("❌ Unhealthy Food")
elif score > 2:
    print("✅ Healthy Food")
else:
    print("⚖️ Moderate / Mixed")

print("\nReasons:")
for r in set(reasons):  # remove duplicates
    print(r)
    
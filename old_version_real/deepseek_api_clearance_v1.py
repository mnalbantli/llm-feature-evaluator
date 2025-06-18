import pandas as pd
import requests
import time

# CSV Dosya Adı
CSV_PATH = "Feature_Review.csv"
API_KEY = "sk-455bbebad6f9481eafbf2d6cdda52f08"  # <-- Buraya kendi API anahtarını koy

# SV'yi oku
df = pd.read_csv(CSV_PATH)
responses = []

# Satır bazlı prompt fonksiyonu
def generate_prompt(row):
    return f"""
You are evaluating a single feature for a dropout prediction model.

COLUMN NAME: {row['Column Name']}
DATA TYPE: {row['Data Type']}
UNIQUE VALUES: {row['Unique Values Count']}
% MISSING: {row['% Missing']}
CARDINALITY: {row['Cardinality']}
RELATED FIELD: {row['Related Field']}
POSSIBLE MEANING: {row['Possible Meaning']}
NOTES: {row['Notes']}

Please complete the following:
1. Is Derived? (Yes/No) + Short Reason
2. Used in Modeling? (Yes/No/Dropped) + Short Reason
3. Transformation Needed? (Yes/No) + Short Reason
4. Time-Dependent? (Yes/No) + Short Reason
5. Bucketed Version Exists? (Yes/No) + Short Reason
6. Interactional Potential? (Yes/No) + Short Reason

Avoid repeating the same reason. Tailor explanations to the actual feature.
Respond ONLY with these 6 outputs in structured format.
    """.strip()

# DeepSeek API ile yanıt alma
def ask_deepseek(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result['choices'][0]['message']['content']

# Satır bazlı işlem
for index, row in df.iterrows():
    print(f"🔍 Processing row {index+1}/{len(df)}: {row['Column Name']}")
    prompt = generate_prompt(row)
    try:
        reply = ask_deepseek(prompt)
        responses.append(reply)
        time.sleep(1)  # API'yı zorlamamak için
    except Exception as e:
        print(f"⚠️ Error in row {index}: {e}")
        responses.append("")

# Cevapları yapılandır
parsed = []
for r in responses:
    entry = {}
    for line in r.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            entry[key.strip()] = val.strip()
    parsed.append(entry)

df_out = pd.concat([df.reset_index(drop=True), pd.DataFrame(parsed)], axis=1)

# Excel olarak kaydet
df_out.to_excel("enriched_feature_review_deepseek.xlsx", index=False)
print("✅ Output saved to enriched_feature_review_deepseek.xlsx")

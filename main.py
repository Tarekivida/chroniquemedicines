import pandas as pd
import openai
import os
import time
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")

if not openai_key or not serper_key:
    raise ValueError("OPENAI_API_KEY or SERPER_API_KEY is missing in the .env file.")

# OpenAI client setup (v1)
client = openai.OpenAI(api_key=openai_key)

# Load input data
df = pd.read_csv("products.csv", sep=';')
df['PRD_NOM'] = df['PRD_NOM'].str.strip().str.upper()

# Convert PRD_EAN13 to consistent string format
df['PRD_EAN13'] = df['PRD_EAN13'].apply(lambda x: str(int(x)) if pd.notnull(x) else "")

# Load or initialize cache
CACHE_FILE = "cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

def get_web_content_for_product(product_name):
    headers = {"X-API-KEY": serper_key}
    params = {"q": f"{product_name} site:.fr"}
    try:
        r = requests.post("https://google.serper.dev/search", json=params, headers=headers)
        result = r.json()
        first_link = result["organic"][0]["link"]

        page = requests.get(first_link, timeout=5)
        soup = BeautifulSoup(page.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs[:5])
        return text.strip()
    except Exception as e:
        print(f"⚠️ Erreur scraping pour {product_name} → {e}")
        return ""

def get_chronicity_score(ean13, product_name):
    if ean13 in cache:
        return cache[ean13]["score"], cache[ean13]["reason"]

    web_content = get_web_content_for_product(product_name)

    prompt = f"""Tu es pharmacien. À partir du nom et du contenu web suivant sur un médicament, attribue un score de chronicité d’achat de 1 à 5. Puis explique ton choix.

Nom du produit : "{product_name}"
Contenu web : "{web_content[:1000]}"

Notation :
1 = Ultra-ponctuel (urgence, 1 seule prise)
2 = Occasionnel / saisonnier (besoin quelques fois par an)
3 = Répété modéré (besoin mensuel ou fréquent mais irrégulier)
4 = Semi-chronique (besoin fréquent, plusieurs fois par mois)
5 = Chronique (usage régulier, traitement de fond)

Format attendu :
Score : [chiffre entre 1 et 5]
Raison : [une phrase expliquant le choix]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        lines = content.splitlines()
        score_line = next((line for line in lines if line.lower().startswith("score")), "")
        reason_line = next((line for line in lines if line.lower().startswith("raison")), "")
        score = int(''.join(filter(str.isdigit, score_line)))
        reason = reason_line.split(":", 1)[1].strip() if ":" in reason_line else ""
        cache[ean13] = {"score": score, "reason": reason}

        print("\n📦 Produit :", product_name)
        print("🔎 Résumé Google :", web_content[:300].replace('\n', ' '))
        print("📈 Score attribué :", score)
        print("🧠 Justification :", reason)

        return score, reason
    except Exception as e:
        print(f"❌ Erreur OpenAI pour {product_name} ({ean13}) → {e}")
        return None, ""

products_to_score = df[['PRD_EAN13', 'PRD_NOM']].drop_duplicates('PRD_EAN13')
products_to_score = products_to_score[~products_to_score['PRD_EAN13'].isin(cache)]

print(f"🧠 Produits à scorer : {len(products_to_score)}")

for _, row in tqdm(products_to_score.iterrows(), total=len(products_to_score)):
    score, reason = get_chronicity_score(row['PRD_EAN13'], row['PRD_NOM'])
    time.sleep(1.2)

# Save cache
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)


# Map scores to main DataFrame with safe EAN key handling
def safe_score(ean):
    if pd.isnull(ean) or not str(ean).strip().isdigit():
        return None
    return cache.get(str(ean), {}).get("score")

def safe_reason(ean):
    if pd.isnull(ean) or not str(ean).strip().isdigit():
        return ""
    return cache.get(str(ean), {}).get("reason", "")

df["CHRONICITY_SCORE"] = df["PRD_EAN13"].map(safe_score)
df["CHRONICITY_REASON"] = df["PRD_EAN13"].map(safe_reason)

df_export = df[df["CHRONICITY_SCORE"].notnull()] \
    .drop_duplicates(subset="PRD_EAN13")[["PRD_EAN13", "PRD_NOM", "CHRONICITY_SCORE", "CHRONICITY_REASON"]]

df_export.columns = ["ean13", "product_name", "score", "justification"]
df_export.to_csv("products_scored.csv", index=False)
print("✅ Fichier 'products_scored.csv' généré avec scores et justifications.")
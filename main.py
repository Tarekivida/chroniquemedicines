import pandas as pd
import openai
import os
import time
import json
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY manquante dans le fichier .env")

# Initialisation du client OpenAI v1
client = openai.OpenAI(api_key=openai_key)

# Lecture du fichier source
df = pd.read_csv("products.csv", sep=';')
df['PRD_NOM'] = df['PRD_NOM'].str.strip().str.upper()

# Chargement du cache (clé = EAN13)
CACHE_FILE = "cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

# Fonction de scoring par EAN13
def get_chronicity_score(ean13, product_name):
    if ean13 in cache:
        return cache[ean13]

    prompt = f"""Voici un nom de produit pharmaceutique : "{product_name}". 
Sur une échelle de 1 à 5, attribue un score de chronicité d’achat basé sur l’usage typique :
1 = ultra-ponctuel (urgence uniquement),
2 = occasionnel ou saisonnier,
3 = modérément répété,
4 = semi-chronique,
5 = chronique (besoin constant).
Réponds uniquement par un chiffre entre 1 et 5. Pas d'explication."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score = int(response.choices[0].message.content.strip()[0])
        cache[ean13] = score
        return score
    except Exception as e:
        print(f"❌ Erreur pour {product_name} ({ean13}) → {e}")
        return None

# Préparation des produits uniques par EAN13
unique_products = df[['PRD_EAN13', 'PRD_NOM']].drop_duplicates()
products_to_score = unique_products[~unique_products['PRD_EAN13'].isin(cache)].head(100)  # Limite à 10 pour test

print(f"🧠 Produits à scorer : {len(products_to_score)}")

# Scoring
for _, row in tqdm(products_to_score.iterrows(), total=len(products_to_score)):
    get_chronicity_score(row['PRD_EAN13'], row['PRD_NOM'])
    time.sleep(1.2)  # Pour éviter le rate limit

# Sauvegarde du cache
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

# Ajout des scores dans le DataFrame d’origine
df["CHRONICITY_SCORE"] = df["PRD_EAN13"].map(cache)

# Export
df.to_csv("products_scored.csv", sep=';', index=False)
print("✅ Fichier 'products_scored.csv' généré avec scores par EAN13.")

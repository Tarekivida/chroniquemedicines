# CLAUDE.md — chroniquesmedicines

Document de reprise à froid. Rédigé le 2026-11-11 après exploration du dépôt (deux branches, historique complet, fichiers locaux non suivis).

## 1. Objectif et contexte

- Scripts Python d'**enrichissement d'un catalogue de produits de pharmacie** (identifiés par **EAN13** + libellé `PRD_NOM`, ex. « CICALFATE+ CR REPAR T/40ML », « DONORMYL CPR SECAB 10 »).
- But principal : attribuer à chaque produit un **score de chronicité d'achat de 1 à 5** via un LLM (OpenAI `gpt-4`) :
  1 = ultra-ponctuel (urgence), 2 = occasionnel/saisonnier, 3 = modérément répété, 4 = semi-chronique, 5 = chronique (traitement de fond).
- But secondaire (`jaccard.py`) : regrouper les libellés en **gammes** (préfixe commun) et **variantes** (suffixe) pour dédoublonner le catalogue.
- Contexte : analyse commerciale officinale (segmenter les ventes entre achats ponctuels et récurrents). L'origine exacte du fichier produits (export LGO, base groupement…) est **inconnue**.
- Utilisateur : l'auteur, en exécution locale ponctuelle (scripts batch, pas d'application).
- Dépôt GitHub : `https://github.com/Tarekivida/chroniquemedicines.git` (attention : **sans « s »** dans le nom du repo distant, contrairement au dossier local).

## 2. Stack

- Python 3 (venv local `.venv/`, version mineure non vérifiée). Pas de `requirements.txt`.
- Paquets installés dans le venv local : `pandas 2.3.1`, `numpy 2.3.2`, `openai 1.97.1` (client v1), `python-dotenv 1.1.1`, `tqdm 4.67.1`, `requests 2.32.4`, `beautifulsoup4 4.13.4`.
- **`joblib` (requis par `jaccard.py`) n'est pas installé** dans le venv local.

## 3. Branches

| Branche | État | Contenu |
|---|---|---|
| `main` (checkée, = `origin/main`, défaut) | Version GitHub la plus récente (commit `5577f66`, 2025-08-22) | `main.py` (scoring « sec » par nom seul, cache EAN→int) + `jaccard.py` (regroupement gamme/variante). |
| `archive-local-2025-07` (locale + `origin/archive-local-2025-07`) | Commit `be8ef18` (2026-11-11), **enfant de `main`** ; archive d'une variante locale de juillet 2025 | Uniquement `main.py` (variante **Serper** : recherche Google + scraping + score **et justification**) et un `.gitignore` « backup defaults ». `jaccard.py` y est **supprimé**. |

Historique `main` : `c5730b0` Initial commit (2025-07-28, incluait `products.csv` et `products_scored.csv`) → `f505b49` → `8c1b08c` « Ignore CSV files » (supprime les CSV du suivi) → `5577f66` (passage à `produits_uniques.csv`, suppression de la limite `.head(100)` de test, ajout de `jaccard.py`).

Chronologie réelle probable : la variante Serper (archive) a été développée localement en juillet 2025 en parallèle de la version GitHub ; elle n'a été poussée qu'au moment de la sauvegarde. Les fichiers locaux `cache.json` et `products_scored.csv` ont été produits **par la variante Serper** (format avec `reason`/`justification`).

## 4. Arborescence (branche main)

```
chroniquesmedicines/
├── main.py      # Scoring de chronicité via gpt-4 (nom produit seul) → products_scored_uniques.csv
├── jaccard.py   # Regroupement par préfixe commun → 3 CSV produits_with_common / grouped / grouped_expanded
└── .gitignore   # .env, cache.json, *.pyc, *.csv, __pycache__/
```

Fichiers locaux non suivis : `.env`, `.venv/`, `cache.json` (50 entrées), `products.csv` (4 623 lignes + en-tête), `products_scored.csv` (49 lignes scorées), `.DS_Store`.

## 5. Fonctionnement détaillé

### 5.1 `main.py` (branche main)
1. `load_dotenv()` ; exige `OPENAI_API_KEY` sinon `ValueError`.
2. Lit **`produits_uniques.csv`** (séparateur `;`, colonnes `PRD_EAN13;PRD_NOM`), normalise `PRD_NOM` (strip + majuscules).
3. Charge `cache.json` (dict EAN13 → score entier).
4. Pour chaque couple (EAN, nom) unique absent du cache : prompt FR à `gpt-4` (`temperature=0`), « réponds uniquement par un chiffre » ; prend le 1er caractère comme score. Pause de 1,2 s entre appels (rate limit).
5. Sauvegarde le cache, ajoute `CHRONICITY_SCORE` à toutes les lignes, écrit **`products_scored_uniques.csv`** (`;`). Le message final mentionne à tort `products_scored.csv`.

### 5.2 `jaccard.py` (branche main)
Malgré son nom, **aucun indice de Jaccard** : regroupement par **préfixe de tokens commun**.
1. Lit `produits_uniques.csv` (`;`), groupe par premier mot de `PRD_NOM`.
2. Dans chaque groupe (parallélisé `joblib`, `n_jobs=-1`), pour chaque produit, prend le préfixe commun (1 ou 2+ tokens) avec le **premier** autre produit qui en partage un → `COMMON_STRING` ; le reste du libellé → `VARIATION`.
3. Si le préfixe est un mot générique (liste `stopwords` : GEL, SERUM, CREME, SPRAY, COMPRIMES…), il est déplacé dans la variation et le mot suivant devient la gamme.
4. Sorties : `produits_with_common.csv`, `produits_grouped.csv` (une ligne par gamme, listes de variantes et EAN, `N_VARIATIONS`), `produits_grouped_expanded.csv`.

### 5.3 `main.py` (branche archive-local-2025-07, variante Serper)
1. Exige `OPENAI_API_KEY` **et** `SERPER_API_KEY` dans `.env`.
2. Lit **`products.csv`** (`;`), normalise les noms, convertit l'EAN en chaîne d'entier (gère les `NaN`).
3. Pour chaque EAN non caché : requête Serper `"<nom> site:.fr"`, télécharge le 1er résultat organique, extrait les 5 premiers `<p>`.
4. Prompt « Tu es pharmacien » avec nom + 1000 caractères de contenu web → réponse au format `Score : n` / `Raison : …` ; parsing ; cache **EAN → {"score", "reason"}**.
5. Export **`products_scored.csv`** (séparateur virgule) avec colonnes `ean13, product_name, score, justification`, uniquement les produits scorés, dédoublonnés par EAN.

## 6. Installation et lancement

```bash
git clone https://github.com/Tarekivida/chroniquemedicines.git chroniquesmedicines && cd chroniquesmedicines
python3 -m venv .venv && source .venv/bin/activate
pip install pandas openai python-dotenv tqdm joblib requests beautifulsoup4
printf 'OPENAI_API_KEY=\nSERPER_API_KEY=\n' > .env   # puis renseigner les valeurs
# Fournir le fichier d'entrée (voir §8), puis :
python main.py        # branche main : nécessite produits_uniques.csv
python jaccard.py     # branche main : nécessite produits_uniques.csv
# Variante Serper (sans checkout) :
git show archive-local-2025-07:main.py > main_serper.py && python main_serper.py   # nécessite products.csv
```

Coût/durée : ~1,2 s + latence API par produit → environ 1h30–2h et plusieurs milliers d'appels `gpt-4` pour ~4 600 lignes (le cache permet de reprendre après interruption, mais il n'est écrit **qu'en fin de boucle** : une interruption perd tout le travail en cours).

Pas de tests, pas de build, pas de déploiement.

## 7. Variables d'environnement

| Variable | Utilisée par | Rôle | Où l'obtenir |
|---|---|---|---|
| `OPENAI_API_KEY` | les deux `main.py` | Appels `gpt-4` | https://platform.openai.com/api-keys |
| `SERPER_API_KEY` | variante archive uniquement | Recherche Google via Serper | https://serper.dev |

Le `.env` local contient ces deux noms (valeurs non recopiées ici). `.env` est ignoré par git.

## 8. Données exclues de git et comment les récupérer

- `*.csv` et `cache.json` sont ignorés.
- **`products.csv`** (entrée de la variante Serper, 4 623 lignes `PRD_EAN13;PRD_NOM`, BOM UTF-8, doublons d'EAN possibles avec libellés différents) : **récupérable depuis l'historique git** (il a été commité puis retiré) :
  `git show c5730b0:products.csv > products.csv`
  Attention : le repo GitHub expose donc ce catalogue dans son historique.
- `products_scored.csv` : la version de l'historique (`git show c5730b0:products_scored.csv`) est celle de la toute première variante (séparateur `;`, colonne `CHRONICITY_SCORE`, ~100 produits scorés max). La version locale (49 produits, format virgule avec justification) provient de la variante Serper et n'est **pas** dans git ; elle se régénère avec la variante Serper.
- **`produits_uniques.csv`** (entrée de `main.py` et `jaccard.py` sur `main`) : **absent localement et jamais commité**. Format attendu : `PRD_EAN13;PRD_NOM`. Probablement une version dédoublonnée de `products.csv` ; sa provenance exacte est **inconnue**. Pour le recréer approximativement :
  `python -c "import pandas as pd; d=pd.read_csv('products.csv',sep=';'); d.drop_duplicates('PRD_EAN13').to_csv('produits_uniques.csv',sep=';',index=False)"`
- `cache.json` : régénéré automatiquement par les scripts (coût API). Le cache local (50 entrées) est au **format de la variante Serper** (`{"score", "reason"}`), incompatible avec `main.py` de `main` (qui attend un entier).

## 9. État, TODO, bugs connus

État : scripts exploratoires ; seuls ~50 produits ont été effectivement scorés (variante Serper). Pas de livrable final identifié.

Bugs / pièges constatés :
- **Types d'EAN incohérents (branche main)** : pandas lit `PRD_EAN13` en entier ; le cache est sauvegardé en JSON (clés converties en chaînes) puis rechargé → `isin(cache)` et `.map(cache)` ne correspondent plus → tout est re-scoré à chaque exécution et `CHRONICITY_SCORE` reste vide pour les entrées venant du cache. La variante archive corrige cela en convertissant l'EAN en chaîne.
- Cache local au format dict incompatible avec la branche main (voir §8).
- `score = int(...[0])` : plante (capturé, renvoie `None`) si le modèle répond autrement qu'un chiffre en premier.
- Le message final de `main.py` annonce `products_scored.csv` alors que le fichier écrit est `products_scored_uniques.csv`.
- `drop_duplicates()` sur (EAN, nom) : un même EAN avec deux libellés est scoré deux fois (la seconde écrase la première dans le cache).
- `jaccard.py` : nom trompeur ; le « meilleur » préfixe est le premier trouvé (dépend de l'ordre des lignes), non le plus long ; `joblib` à installer.
- Variante Serper : premier résultat Google seulement, `timeout=5`, pas de User-Agent ; échecs silencieux → score sur nom seul.
- `gpt-4` est un modèle ancien et coûteux ; un modèle plus récent/moins cher conviendrait probablement.
- Cache écrit uniquement à la fin : prévoir des sauvegardes périodiques.

## 10. Conventions

- CSV d'entrée en `;` (export français) ; sortie Serper en `,`.
- Prompts et messages console en français, avec emojis.
- Clé du cache = EAN13 (idéalement en chaîne).

## 11. Reprendre le projet

1. Cloner ; recréer le venv et installer les dépendances (§6) ; créer `.env` avec les deux clés.
2. Restaurer les données : `git show c5730b0:products.csv > products.csv`, puis créer `produits_uniques.csv` (§8) si l'on veut utiliser la branche main.
3. Choisir la variante : **Serper (archive)** = plus précise (contexte web + justification), plus lente/chère ; **main** = rapide mais naïve et boguée sur les types d'EAN.
4. Fusionner le meilleur des deux : EAN en chaîne, cache dict `{score, reason}`, sauvegarde périodique du cache, modèle OpenAI récent, `requirements.txt`.
5. Tester sur 20 produits (`.head(20)`) avant un run complet.

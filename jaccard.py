import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Load the file
df = pd.read_csv("produits_uniques.csv", sep=";")

# Liste des stopwords génériques
stopwords = {
    "GEL", "SERUM", "TROUSSE", "COUSSIN", "VERNIS",
    "SPRAY", "MASQUE", "CREME", "HUILE", "SHAMPOOING",
    "NEUT", "POMMADE", "CAPSULES", "COMPRIMES"
}

def common_prefix_tokens(a: str, b: str) -> str:
    ta, tb = a.upper().split(), b.upper().split()
    prefix = []
    for x, y in zip(ta, tb):
        if x == y:
            prefix.append(x)
        else:
            break
    if len(prefix) >= 2:
        return " ".join(prefix)
    elif len(prefix) == 1:
        return prefix[0]
    else:
        return ""

def process_group(group):
    results = []
    rows = group.to_dict("records")
    for row in rows:
        name = str(row["PRD_NOM"])
        best_match = ""
        for row2 in rows:
            if row["PRD_EAN13"] == row2["PRD_EAN13"]:
                continue
            candidate = common_prefix_tokens(name, str(row2["PRD_NOM"]))
            if candidate:
                best_match = candidate
                break

        variation = name
        if best_match and name.upper().startswith(best_match):
            variation = name[len(best_match):].strip()

        # 🟢 Si best_match est un stopword → on prend le premier mot significatif de la variation
        if best_match in stopwords:
            tokens = variation.split()
            if tokens:
                new_common = tokens[0].upper()
                new_variation = f"{best_match} {variation}"
                best_match = new_common
                variation = new_variation

        results.append((row["PRD_EAN13"], name, best_match, variation))
    return results

# Group by first token
df["FIRST_TOKEN"] = df["PRD_NOM"].str.split().str[0]
groups = [g for _, g in df.groupby("FIRST_TOKEN")]

# Run in parallel
all_results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_group)(g) for g in groups
)
flat = [item for sublist in all_results for item in sublist]

# Map back
common_map = {ean: (common, var) for ean, name, common, var in flat}
df["COMMON_STRING"] = df["PRD_EAN13"].map(lambda x: common_map.get(x, ("", ""))[0])
df["VARIATION"] = df["PRD_EAN13"].map(lambda x: common_map.get(x, ("", ""))[1])

# Save full dataset
df.to_csv("produits_with_common.csv", sep=";", index=False)

# Grouped version with count of variations
grouped = df.groupby("COMMON_STRING").agg({
    "VARIATION": lambda x: list(set(x)),
    "PRD_EAN13": lambda x: list(set(x))
}).reset_index()

grouped["N_VARIATIONS"] = grouped["VARIATION"].apply(len)

grouped.to_csv("produits_grouped.csv", sep=";", index=False)

# Expanded version
rows = []
for common, sub in df.groupby("COMMON_STRING"):
    for var, ean in zip(sub["VARIATION"], sub["PRD_EAN13"]):
        rows.append((common, ean, var))
grouped_expanded = pd.DataFrame(rows, columns=["COMMON_STRING", "EAN13", "VARIATION"])
grouped_expanded.to_csv("produits_grouped_expanded.csv", sep=";", index=False)

print("✅ Generated:")
print(" - produits_with_common.csv (full rows)")
print(" - produits_grouped.csv (one row per COMMON_STRING, variations+EANs+N_VARIATIONS)")
print(" - produits_grouped_expanded.csv (one row per COMMON_STRING + EAN + VARIATION)")

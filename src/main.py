# Installér biblioteker først (kør i terminal hvis ikke installeret)
# pip install pandas scikit-learn shap tqdm

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from scipy.sparse import vstack
import shap
from tqdm import tqdm

# 1) Find script-mappen og indlæs CSV
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path  = os.path.join(base_path, "emails.csv")
df = pd.read_csv(csv_path)

# 2) Rens kolonner
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 3) Tjek at de nødvendige kolonner findes
if not {'text_combined', 'label'}.issubset(df.columns):
    print("FEJL: CSV skal have kolonnerne 'text_combined' og 'label'")
    sys.exit(1)

# 4) Håndter label-kolonnen (0/1 eller string der kan castes)
if df['label'].dtype == object:
    # f.eks. strings "0" og "1" eller tal som tekst
    df['label'] = df['label'].astype(str).str.strip().astype(int)
else:
    df['label'] = df['label'].astype(int)

# 5) Drop rækker med andre værdier end 0/1
df = df[df['label'].isin([0,1])].copy()
if df.empty:
    print("FEJL: Ingen rækker med label 0 eller 1 fundet")
    sys.exit(1)

# 6) Fill na og drop helt tomme mails
df['text_combined'] = df['text_combined'].fillna("").astype(str)
non_empty = df['text_combined'].str.strip().astype(bool).sum()
if non_empty == 0:
    print("FEJL: Ingen ikke-tomme mails fundet i 'text_combined'")
    sys.exit(1)
df = df[df['text_combined'].str.strip().astype(bool)].copy()

# 7) Forbered features og labels
X_raw = df['text_combined']
y     = df['label']

# 8) TF-IDF-vektorisering med begrænsning
vectorizer = TfidfVectorizer(
    max_features=1000,  # kun de 100 mest informative ord
    min_df=1,
    max_df=0.9,
    stop_words='english'
)
try:
    X = vectorizer.fit_transform(X_raw)
except ValueError as e:
    print("FEJL under TF-IDF (tom ordbog):", e)
    sys.exit(1)

print("Ordbogsstørrelse:", len(vectorizer.vocabulary_))

# 9) Split i træning (60%), validering (20%), test (20%)
X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
    X, y, df, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
    X_temp, y_temp, df_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# 10) Definér tre modeller
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM':            LinearSVC(random_state=42),
    'KNN':            KNeighborsClassifier(n_neighbors=5)
}

# 11) Træn og evaluér på validerings-sættet
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    print(classification_report(y_val, y_pred_val, target_names=['Safe', 'Phishing']))

# 12) Gen-træn Random Forest på train+val
best_model = RandomForestClassifier(random_state=42)
X_trainval = vstack([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
best_model.fit(X_trainval, y_trainval)

# 13) Endelig evaluation på test-sættet
y_pred_test = best_model.predict(X_test)
print("\n=== Endelige test-resultater (Random Forest) ===")
print(classification_report(y_test, y_pred_test, target_names=['Safe', 'Phishing']))

# 14) SHAP forklaringer (med beskyttelse mod out-of-bounds)
X_test_dense = X_test.toarray()
sample_size  = min(100, X_test_dense.shape[0])
X_sample     = X_test_dense[:sample_size]
df_sample    = df_test.iloc[:sample_size].copy()

print(f"\nBeregner SHAP-værdier for {sample_size} e-mails…")
explainer      = shap.TreeExplainer(best_model)
shap_values_all = []

for xi in tqdm(X_sample, desc="SHAP forklaringer"):
    sv = explainer.shap_values(xi.reshape(1, -1), check_additivity=False)
    arr = sv[1] if isinstance(sv, list) else sv
    shap_values_all.append(arr.reshape(-1,))

feature_names = vectorizer.get_feature_names_out()
n_feat       = feature_names.shape[0]

def get_top_shap_features(shap_arr, features, n=5):
    # Klip til feature-længde hvis nødvendigt
    if shap_arr.shape[0] > features.shape[0]:
        shap_arr = shap_arr[:features.shape[0]]
    idxs = np.argsort(shap_arr)[-n:][::-1]
    # Filtrér sikre indekser
    idxs = [i for i in idxs if 0 <= i < features.shape[0]]
    return [(features[i], float(shap_arr[i])) for i in idxs]

df_sample['SHAP_explanation'] = [
    get_top_shap_features(shap_values_all[i], feature_names, n=5)
    for i in range(sample_size)
]

# 15) Gem sample med forklaringer til CSV
output_csv = os.path.join(base_path, 'emails_med_SHAP_samples.csv')
df_sample.to_csv(output_csv, index=False)
print(f"SHAP-forklaringer gemt i '{output_csv}'")

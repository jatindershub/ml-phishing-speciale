
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
if df['text_combined'].str.strip().astype(bool).sum() == 0:
    print("FEJL: Ingen ikke-tomme mails fundet i 'text_combined'")
    sys.exit(1)
df = df[df['text_combined'].str.strip().astype(bool)].copy()

# 7) Forbered features og labels
X_raw = df['text_combined']
y     = df['label']

# 8) TF-IDF-vektorisering med begrænsning
vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=1,
    max_df=0.9,
    stop_words='english'
)
try:
    X = vectorizer.fit_transform(X_raw)
except ValueError as e:
    print("FEJL under TF-IDF (tom lexicon):", e)
    sys.exit(1)

print("Lexicon:", len(vectorizer.vocabulary_))

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

# 11) Træn, evaluér, log og find bedste model ud fra F1 på valideringssættet
best_score = -1.0
best_name  = None

val_report_path = os.path.join(base_path, 'val_classification_reports.txt')
with open(val_report_path, 'w') as f:
    for name, model in models.items():
        f.write(f"=== {name} ===\n")
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)

        # Classification report
        report = classification_report(
            y_val, y_pred_val,
            target_names=['Safe', 'Phishing'],
            digits=4
        )
        print(report)
        f.write(report + "\n")

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_val, labels=[1, 0])
        print("Confusion Matrix (rows=Actual, cols=Predicted):")
        print("           Predicted Phish  Predicted Safe")
        print(f"Actual Phish    {cm[0,0]:5d}              {cm[0,1]:5d}")
        print(f"Actual Safe     {cm[1,0]:5d}              {cm[1,1]:5d}\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")

        # Beregn F1 for at vælge 'bedste'
        score = f1_score(y_val, y_pred_val)
        if score > best_score:
            best_score = score
            best_name  = name

print(f"\nValgt bedste model: {best_name} (F1={best_score:.4f})")

# 12) Gen-træn den bedste model på train+val
if best_name == 'Random Forest':
    best_model = RandomForestClassifier(random_state=42)
elif best_name == 'SVM':
    best_model = LinearSVC(random_state=42)
else:
    best_model = KNeighborsClassifier(n_neighbors=5)

X_trainval = vstack([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
best_model.fit(X_trainval, y_trainval)

# 13) Endelig evaluation på test-sættet
y_pred_test = best_model.predict(X_test)
final_report = classification_report(
    y_test, y_pred_test,
    target_names=['Safe', 'Phishing'],
    digits=4
)
print(f"\n=== Endelige test-resultater ({best_name}) ===")
print(final_report)

test_report_path = os.path.join(base_path, 'final_test_classification_report.txt')
with open(test_report_path, 'w') as f:
    f.write(f"=== Endelige test-resultater ({best_name}) ===\n")
    f.write(final_report)

# 14) Test nyt datasæt: emails2.csv
csv2_path = os.path.join(base_path, "emails2.csv")
df2 = pd.read_csv(csv2_path)

# 14.1) Rens kolonner: behold kun 'feature' og 'label'
df2 = df2[['feature', 'label']].copy()

# 14.2) Omdøb for genbrug
df2 = df2.rename(columns={'feature': 'text_combined'})

# 14.3) Map label-strings til 0/1
label_map = {'legitimate': 0, 'phishing': 1}
df2['label'] = df2['label'].astype(str).str.strip().map(label_map)

# 14.4) Drop evt. ugyldige labels
df2 = df2[df2['label'].isin([0,1])]
if df2.empty:
    print("FEJL: Ingen rækker med gyldige labels i emails2.csv")
    sys.exit(1)

# 14.5) Sørg for ingen tomme mails
df2['text_combined'] = df2['text_combined'].fillna("").astype(str)
df2 = df2[df2['text_combined'].str.strip().astype(bool)]
if df2.empty:
    print("FEJL: Ingen ikke-tomme mails i emails2.csv")
    sys.exit(1)

# 14.6) Lav features og labels for nyt datasæt
X2_raw = df2['text_combined']
y2     = df2['label']

# 14.7) TF-IDF-transform (brug allerede fit’ede vectorizer)
X2 = vectorizer.transform(X2_raw)

# 14.8) Predict med best_model
y2_pred = best_model.predict(X2)

# 14.9) Evaluer og print
print("\n=== Test på emails2.csv ===")
print(classification_report(y2, y2_pred, target_names=['Legitimate','Phishing'], digits=4))

cm2 = confusion_matrix(y2, y2_pred, labels=[1,0])
print("Confusion Matrix (rows=Actual, cols=Predicted):")
print("           Predicted Phish  Predicted Legit")
print(f"Actual Phish    {cm2[0,0]:5d}              {cm2[0,1]:5d}")
print(f"Actual Legit    {cm2[1,0]:5d}              {cm2[1,1]:5d}")

# (Valgfrit) Gem resultaterne til fil
test2_report_path = os.path.join(base_path, 'test2_classification_report.txt')
with open(test2_report_path, 'w') as f:
    f.write("=== Test på emails2.csv ===\n")
    f.write(classification_report(y2, y2_pred, target_names=['Legitimate','Phishing'], digits=4))
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm2))

# 15) SHAP forklaringer (med beskyttelse mod out-of-bounds)
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
    if shap_arr.shape[0] > features.shape[0]:
        shap_arr = shap_arr[:features.shape[0]]
    idxs = np.argsort(shap_arr)[-n:][::-1]
    idxs = [i for i in idxs if 0 <= i < features.shape[0]]
    return [(features[i], float(shap_arr[i])) for i in idxs]

df_sample['SHAP_explanation'] = [
    get_top_shap_features(shap_values_all[i], feature_names, n=5)
    for i in range(sample_size)
]

# 16) Gem sample med forklaringer til CSV
output_csv = os.path.join(base_path, 'emails_med_SHAP_samples.csv')
df_sample.to_csv(output_csv, index=False)
print(f"SHAP-forklaringer gemt i '{output_csv}'")

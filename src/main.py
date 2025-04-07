# Installér biblioteker først (kør i terminal hvis ikke installeret)
# pip install pandas scikit-learn shap tqdm

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from scipy.sparse import vstack
import shap
from tqdm import tqdm
import numpy as np

# Find mappen hvor scriptet ligger
base_path = os.path.dirname(os.path.abspath(__file__))

# Lav absolut sti til emails.csv i samme mappe
csv_path = os.path.join(base_path, "emails.csv")

# Indlæs data
df = pd.read_csv(csv_path)

# Fjern unødvendige kolonner (første tom kolonne)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Konvertér labels til binære værdier
df['label'] = df['Email Type'].map({"Safe Email": 0, "Phishing Email": 1})

# Rens for tomme e-mails
df['Email Text'] = df['Email Text'].fillna("")

# Features og labels
X_raw = df['Email Text']
y = df['label']

# TF-IDF vektorisering
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X = vectorizer.fit_transform(X_raw)

# Split i træning (60%), validering (20%), test (20%)
X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
    X, y, df, test_size=0.4, stratify=y, random_state=42)

X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
    X_temp, y_temp, df_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Modellerne
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': LinearSVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Træn og evaluer modeller
for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    print(classification_report(y_val, y_pred_val, target_names=['Safe', 'Phishing']))

# Random Forest som bedste model
best_model = RandomForestClassifier(random_state=42)
X_trainval = vstack([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
best_model.fit(X_trainval, y_trainval)

# Test på testdata
y_pred_test = best_model.predict(X_test)
print("\nEndelig test-resultater (Random Forest):")
print(classification_report(y_test, y_pred_test, target_names=['Safe', 'Phishing']))

# Konverter sparse matrix til dense for SHAP
X_test_dense = X_test.toarray()

# Brug kun et udsnit for hurtigere test
sample_size = 100  # ← ændr dette tal for at tage flere eller færre e-mails
X_sample = X_test_dense[:sample_size]
df_sample = df_test.iloc[:sample_size]

# SHAP forklaringer
print(f"\nBeregner SHAP-værdier for {sample_size} e-mails med {X_sample.shape[1]} features...")
explainer = shap.TreeExplainer(best_model)

shap_values_all = []
for i in tqdm(range(sample_size), desc="SHAP forklaringer"):
    shap_vals = explainer.shap_values(X_sample[i:i+1], check_additivity=False)
    if isinstance(shap_vals, list):  # fx [class_0, class_1]
        shap_values_all.append(shap_vals[1])  # phishing-klassen
    else:
        shap_values_all.append(shap_vals)  # regression eller binær uden class-split

# Kun forklaringer for klassen "Phishing"
shap_values_phishing = shap_values_all  # hver sv er allerede én vektor
feature_names = vectorizer.get_feature_names_out()

# SHAP top-5 ord pr. e-mail
def get_top_shap_features(shap_vals, features, n=5):
    max_index = len(features)
    top_indices = shap_vals.argsort()[-n:][::-1]
    top_indices = [i for i in top_indices if i < max_index]  # filter ud af bounds
    return [(features[i], shap_vals[i]) for i in top_indices]

df_sample = df_sample.copy()
df_sample['SHAP_explanation'] = [
    get_top_shap_features(shap_values_phishing[i].flatten(), feature_names)
    for i in range(sample_size)
]

# Gem til ny CSV-fil med SHAP-forklaringer
df_sample.to_csv('emails_with_SHAP_sample.csv', index=False)
print("\nSHAP forklaringer (sample) gemt i 'emails_with_SHAP_sample.csv'")

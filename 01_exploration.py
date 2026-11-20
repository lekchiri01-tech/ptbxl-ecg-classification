import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

DATA_PATH = Path(r"C:\Users\Victus\Desktop\ptb-xl-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")  # <- ton chemin ici

df = pd.read_csv(DATA_PATH / "ptbxl_database.csv", index_col="ecg_id")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)


# %% Étape 1: un ECG normal, dérivation II, zoom sur 3 secondes


ids_normaux = df[df["scp_codes"].apply(lambda d: "NORM" in d)].index
ecg_id = ids_normaux[0]
row = df.loc[ecg_id]

signal, meta = wfdb.rdsamp(str(DATA_PATH / row["filename_lr"]))
fs = meta["fs"]
derivation = meta["sig_name"].index("II")  # la plus utilisée pour lire le rythme

n = 3 * fs  # 3 secondes
t = np.arange(n) / fs

plt.figure(figsize=(14, 4))
plt.plot(t, signal[:n, derivation])
plt.title(f"ECG normal n°{ecg_id}, dérivation II, 3 premières secondes")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude (mV)")
plt.grid(alpha=0.3)
plt.savefig("etape1_ecg_normal.png", dpi=150)
plt.show()


#%%Étape 2: dérivations, fréquence déchantillonnage, unités

print("Nombre de points:", signal.shape[0])
print("Nombre de dérivations:", signal.shape[1])
print("Noms des dérivations:", meta["sig_name"])
print("Unités:", meta["units"])
print("Fréquence d'échantillonnage:", fs, "Hz")
print("Durée:", signal.shape[0] / fs, "secondes")

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for ax, nom in zip(axes, ["I", "II", "V1"]):
    idx = meta["sig_name"].index(nom)
    ax.plot(t, signal[:n, idx])
    ax.set_ylabel(f"{nom} (mV)")
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Temps (s)")
fig.suptitle(f"ECG n°{ecg_id}: trois dérivations, mêmes 3 secondes")
plt.tight_layout()
plt.savefig("etape2_derivations.png", dpi=150)
plt.show()


# %% Étape 3: les étiquettes de diagnostic


print("\n--- Exemples de codes de diagnostic (code: certitude en %) ---")
print(df["scp_codes"].head(10))

print("\n--- Exemples de rapports écrits par les médecins ---")
print(df["report"].head(5))

# Table qui relie chaque code à une grande classe
agg_df = pd.read_csv(DATA_PATH / "scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df["diagnostic"] == 1]  # on garde les vrais diagnostics

def agreger(codes):
    classes = set()
    for code in codes:
        if code in agg_df.index:
            classes.add(agg_df.loc[code, "diagnostic_class"])
    return list(classes)

df["classes"] = df["scp_codes"].apply(agreger)

print("\n--- Codes et classes associées ---")
print(df[["scp_codes", "classes"]].head(10))
print("\nECG sans aucune classe:", (df["classes"].str.len() == 0).sum())
print("ECG avec plusieurs classes:", (df["classes"].str.len() > 1).sum())
# %% Étape 4: combien d'ECG par classe
from collections import Counter

compteur = Counter(c for liste in df["classes"] for c in liste)
repartition = pd.Series(compteur).sort_values(ascending=False)

print("\n--- Nombre d'ECG par classe ---")
print(repartition)
print("\nPourcentage:")
print((repartition / len(df) * 100).round(1))

plt.figure(figsize=(9, 5))
repartition.plot(kind="bar")
plt.title("PTB-XL: nombre d'ECG par classe diagnostique")
plt.xlabel("Classe")
plt.ylabel("Nombre d'ECG")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("etape4_repartition_classes.png", dpi=150)
plt.show()

# %% Convertir les classes en nombres (0 ou 1)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

for c in CLASSES:
    df[c] = df["classes"].apply(lambda liste: 1 if c in liste else 0)

print(df[["classes"] + CLASSES].head(10))
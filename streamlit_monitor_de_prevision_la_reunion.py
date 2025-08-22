# app.py — Monitor de prévision conso/production & alertes (La Réunion)
# ---------------------------------------------------------------
# Ce prototype Streamlit répond au cahier des charges :
#  - Formulaire opérateur (nom, fonction, département)
#  - Saisie des variables utilisées dans l’étude (météo, démographie, événements, prod. par source)
#  - Prévision conso & prod. simultanée (démo), détection du croisement et délai avant déficit
#  - Reco/alerte : ressource à mobiliser (batteries → thermique → effacement)
#  - Basculer entre tarifs (Bleu / Vert) et estimer le coût du déficit
#  - Fenêtre temporelle “glissante” (horizon min/heure + lissage)
#  - Classement zones énergivores (import CSV ou saisie rapide)
#  - Hooks pour charger un vrai modèle (LSTM/XGBoost) ayant R²≈0.95 (voir TODO)
#
# NOTE IMPORTANTE :
#  Ce fichier inclut un modèle de démonstration (MockModel) afin que l’interface soit utilisable
#  tout de suite. Remplacez-le par votre modèle entraîné (LSTM .h5 / XGBoost .pkl) dans la
#  section "=== CHARGEMENT DU MODELE ===". Tous les points d’intégration sont prêts.
# ---------------------------------------------------------------

import io
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================
#           CONFIG
# =============================
st.set_page_config(
    page_title="Monitor Énergie – Prévision & Alertes",
    page_icon="⚡",
    layout="wide",
)

# --------- Helpers ---------
@dataclass
class Tarifs:
    nom: str
    prix_kwh_hp: float
    prix_kwh_hc: float

# Valeurs par défaut (à ajuster en prod)
TARIF_BLEU = Tarifs("Bleu", prix_kwh_hp=0.185, prix_kwh_hc=0.155)
TARIF_VERT = Tarifs("Vert", prix_kwh_hp=0.145, prix_kwh_hc=0.115)


# =============================
#   MOCK MODEL (à remplacer)
# =============================
class MockModel:
    """Petit modèle démonstratif (remplacer par LSTM/XGB réel).
    Il combine météo + démographie + événements + inertie de série.
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.coefs = {
            "temp": 12_000,
            "humid": 2_000,
            "pluie": -1_500,
            "soleil": 1_800,
            "pop": 0.9,
            "event": 85_000,
        }
        self.bias = 2_600_000
        self.noise = lambda n: rng.normal(0, 9_000, size=n)

    def predict_consumption(self, df: pd.DataFrame) -> np.ndarray:
        base = (
            self.bias
            + self.coefs["temp"] * (df["temperature"] - 24)
            + self.coefs["humid"] * (df["humidite"] - 70) / 10
            + self.coefs["pluie"] * (df["precipitations"]) / 10
            + self.coefs["soleil"] * (df["ensoleillement"]) / 10
            + self.coefs["pop"] * (df["population"] - 800_000) / 10
            + self.coefs["event"] * (df["evenement"])
        )
        trend = 3_000 * np.arange(len(df)) / max(1, len(df) - 1)
        return np.maximum(0, base + trend + self.noise(len(df)))

    def predict_production(self, df: pd.DataFrame) -> np.ndarray:
        # Conversion GWh → kWh pour garder la même unité
        prod_kwh = (
            (df["prod_fossiles_gwh"]
             + df["prod_solaire_gwh"]
             + df["prod_hydraulique_gwh"]
             + df["prod_eolien_gwh"]
             + df["prod_biomasse_gwh"]
             + df["prod_biogaz_gwh"]) * 1e6
        )
        # Ajoute de l’inertie + petite variabilité
        wiggle = np.sin(np.linspace(0, 3*np.pi, len(df))) * 25_000
        return np.maximum(0, prod_kwh + wiggle)


# =============================
#     CHARGEMENT DU MODELE
# =============================
@st.cache_resource
def load_model():
    # TODO (remplacer) :
    #  - LSTM : from tensorflow.keras.models import load_model ; model = load_model("modele_lstm.h5")
    #  - XGB  : import pickle ; model = pickle.load(open("modele_xgb.pkl","rb"))
    return MockModel()

model = load_model()


# =============================
#     BARRE LATERALE – PARAMS
# =============================
with st.sidebar:
    st.header("⚙️ Paramètres globaux")
    tarif_nom = st.radio("Profil tarifaire", ["Tarif Bleu", "Tarif Vert"], index=0)

    col_tar1, col_tar2 = st.columns(2)
    if tarif_nom == "Tarif Bleu":
        tarifs = TARIF_BLEU
    else:
        tarifs = TARIF_VERT
    with col_tar1:
        prix_hp = st.number_input("Prix HP (€/kWh)", value=tarifs.prix_kwh_hp, step=0.001)
    with col_tar2:
        prix_hc = st.number_input("Prix HC (€/kWh)", value=tarifs.prix_kwh_hc, step=0.001)

    st.markdown("---")
    st.subheader("⏱️ Fenêtre & lissage")
    horizon_unit = st.selectbox("Unité d’horizon", ["minutes", "heures"], index=1)
    max_h = 24 if horizon_unit == "heures" else 180
    horizon = st.slider(f"Horizon de prévision ({horizon_unit})", 5, max_h, value=min(6, max_h))
    roll_win = st.slider("Lissage (moy. mobile – pas)", 1, 12, value=3)

    st.markdown("---")
    st.subheader("📦 Capacités mobilisables")
    cap_batterie_mwh = st.number_input("Capacité batteries (MWh)", min_value=0.0, value=35.0, step=1.0)
    cap_thermique_mwh = st.number_input("Capacité thermique secours (MWh)", min_value=0.0, value=120.0, step=5.0)
    effacement_mwh = st.number_input("Effacement (MWh)", min_value=0.0, value=15.0, step=1.0)


# =============================
#   ENTETE & FORMULAIRE OPS
# =============================
st.title("⚡ Monitor de Prévision – La Réunion")
st.caption("Prévoir conso/production, alerter les croisements et recommander les bascules.")

with st.expander("👤 Formulaire opérateur (obligatoire pour valider une prévision)", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        nom = st.text_input("Nom et Prénom")
    with colB:
        fonction = st.text_input("Fonction")
    with colC:
        departement = st.text_input("Département / Entité")

# =============================
#      SAISIE DES VARIABLES
# =============================
st.subheader("🧮 Variables d’entrée (instant T)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    temperature = st.number_input("Température (°C)", value=26.0)
    humidite = st.number_input("Humidité (%)", value=78.0)
with col2:
    precipitations = st.number_input("Précipitations (mm)", value=2.0)
    ensoleillement = st.number_input("Ensoleillement (h)", value=7.0)
with col3:
    population = st.number_input("Population (Saint-Denis)", value=860_000, step=1000)
    evenement = st.selectbox("Événement en cours", ["Aucun", "Fête/Jour férié", "Événement culturel"], index=0)
with col4:
    date_ref = st.date_input("Date de référence", value=dt.date.today())
    heure_ref = st.time_input("Heure de référence", value=dt.datetime.now().time())

st.markdown("**Production par source (GWh)** – valeurs instantanées/projetées")
colp1, colp2, colp3, colp4, colp5, colp6 = st.columns(6)
with colp1:
    prod_fossiles_gwh = st.number_input("Fossiles", value=1.335, step=0.01)
with colp2:
    prod_solaire_gwh = st.number_input("Solaire", value=0.293, step=0.01)
with colp3:
    prod_hydraulique_gwh = st.number_input("Hydraulique", value=0.394, step=0.01)
with colp4:
    prod_eolien_gwh = st.number_input("Éolien", value=0.018, step=0.001)
with colp5:
    prod_biomasse_gwh = st.number_input("Biomasse", value=1.041, step=0.01)
with colp6:
    prod_biogaz_gwh = st.number_input("Biogaz", value=0.015, step=0.001)

# =============================
#    PREPARATION DU SCENARIO
# =============================
if horizon_unit == "heures":
    freq = "H"
    steps = horizon
else:
    freq = "T"
    steps = horizon

start_dt = dt.datetime.combine(date_ref, heure_ref)
idx = pd.date_range(start=start_dt, periods=steps, freq=freq)

# Construire DataFrame des features
features = pd.DataFrame({
    "timestamp": idx,
    "temperature": np.linspace(temperature, temperature, steps),
    "humidite": np.linspace(humidite, humidite, steps),
    "precipitations": np.linspace(precipitations, precipitations, steps),
    "ensoleillement": np.linspace(ensoleillement, max(0, ensoleillement-0.1*steps), steps),
    "population": population,
    "evenement": 0 if evenement == "Aucun" else 1,
    "prod_fossiles_gwh": np.linspace(prod_fossiles_gwh, prod_fossiles_gwh*0.98, steps),
    "prod_solaire_gwh": np.linspace(prod_solaire_gwh, max(0, prod_solaire_gwh*0.95), steps),
    "prod_hydraulique_gwh": np.linspace(prod_hydraulique_gwh, prod_hydraulique_gwh, steps),
    "prod_eolien_gwh": np.linspace(prod_eolien_gwh, prod_eolien_gwh*1.05, steps),
    "prod_biomasse_gwh": np.linspace(prod_biomasse_gwh, prod_biomasse_gwh, steps),
    "prod_biogaz_gwh": np.linspace(prod_biogaz_gwh, prod_biogaz_gwh, steps),
})

# =============================
#        INFERENCE DEMO
# =============================
pred_conso = model.predict_consumption(features)
pred_prod = model.predict_production(features)

# Lissage optionnel
if roll_win > 1:
    pred_conso = pd.Series(pred_conso).rolling(roll_win, min_periods=1).mean().values
    pred_prod = pd.Series(pred_prod).rolling(roll_win, min_periods=1).mean().values

res = pd.DataFrame({
    "timestamp": features["timestamp"],
    "Consommation (kWh)": pred_conso,
    "Production (kWh)": pred_prod,
})
res["Delta (kWh) = Prod - Conso"] = res["Production (kWh)"] - res["Consommation (kWh)"]

# Détection du premier croisement (déficit)
deficit_mask = res["Delta (kWh) = Prod - Conso"] < 0
first_deficit_idx = np.argmax(deficit_mask.values) if deficit_mask.any() else None

# =============================
#      VISU – COURBES & KPI
# =============================
colg1, colg2 = st.columns([2, 1])
with colg1:
    st.subheader("📈 Courbes conso vs prod")
    fig = px.line(
        res, x="timestamp", y=["Consommation (kWh)", "Production (kWh)"],
        labels={"timestamp": "Temps"},
        title="Prévision à court terme",
    )
    st.plotly_chart(fig, use_container_width=True)

with colg2:
    st.subheader("🔎 Indicateurs")
    conso_tot = float(res["Consommation (kWh)"].sum())
    prod_tot = float(res["Production (kWh)"].sum())
    delta_tot = prod_tot - conso_tot

    st.metric("Énergie prévue – Conso", f"{conso_tot:,.0f} kWh")
    st.metric("Énergie prévue – Prod", f"{prod_tot:,.0f} kWh")
    st.metric("Solde (Prod – Conso)", f"{delta_tot:,.0f} kWh")

# =============================
#     ALERTES & RECOMMANDATIONS
# =============================
st.markdown("---")
st.subheader("🚨 Alertes et recommandations")

if first_deficit_idx is not None and deficit_mask.any():
    t0 = res.iloc[first_deficit_idx]["timestamp"]
    manque_kwh = abs(float(res.iloc[first_deficit_idx:]["Delta (kWh) = Prod - Conso"].sum()))
    # Coût indicatif (HP/HC supposé 50/50 sur l’horizon)
    cout_unitaire = 0.5 * prix_hp + 0.5 * prix_hc
    cout_deficit = manque_kwh * cout_unitaire

    # Plan de mobilisation
    reste = manque_kwh / 1000.0  # kWh → MWh
    plan = []
    mobil_batt = min(reste, cap_batterie_mwh); reste -= mobil_batt
    if mobil_batt > 0:
        plan.append(("Batteries", mobil_batt))
    mobil_th = min(max(0.0, reste), cap_thermique_mwh); reste -= mobil_th
    if mobil_th > 0:
        plan.append(("Thermique", mobil_th))
    mobil_eff = min(max(0.0, reste), effacement_mwh); reste -= mobil_eff
    if mobil_eff > 0:
        plan.append(("Effacement", mobil_eff))

    st.error(f"Croisement prévu à partir de **{t0:%d/%m %H:%M}** → déficit global ≈ **{manque_kwh:,.0f} kWh**.")
    st.write(f"Coût indicatif (profil **{tarif_nom}**) ≈ **{cout_deficit:,.0f} €** sur l’horizon.")

    if plan:
        st.markdown("**Plan de bascule recommandé (ordre)** :")
        for nom, mwh in plan:
            st.write(f"- {nom} : **{mwh:.1f} MWh**")
    if reste > 0:
        st.warning(f"Risque résiduel de **{reste:.1f} MWh** après mobilisation → affiner horizon ou variables.")
else:
    st.success("Aucun croisement (déficit) détecté sur l’horizon. ✅")

# =============================
#   ZONES LES PLUS ÉNERGIVORES
# =============================
st.markdown("---")
st.subheader("🗺️ Zones énergivores")

upload = st.file_uploader("Importer un CSV des zones (colonnes: zone, consommation_kwh)")
if "zones_df" not in st.session_state:
    st.session_state.zones_df = pd.DataFrame({
        "zone": ["Nord", "Est", "Ouest", "Sud"],
        "consommation_kwh": [820_000, 540_000, 1_120_000, 660_000],
    })

if upload is not None:
    try:
        st.session_state.zones_df = pd.read_csv(upload)
    except Exception as e:
        st.warning(f"CSV invalide ({e}). Utilisation des valeurs par défaut.")

zones = st.session_state.zones_df.copy()
zones = zones.sort_values("consommation_kwh", ascending=False)
colz1, colz2 = st.columns([1.2, 1])
with colz1:
    zfig = px.bar(zones, x="zone", y="consommation_kwh", title="Classement zones énergivores")
    st.plotly_chart(zfig, use_container_width=True)
with colz2:
    top_zone, top_val = zones.iloc[0]["zone"], float(zones.iloc[0]["consommation_kwh"])
    st.metric("Zone la plus énergivore", f"{top_zone}", delta=f"{top_val:,.0f} kWh")

# =============================
#     EXPORT / PROCES-VERBAL
# =============================
st.markdown("---")
colx1, colx2 = st.columns(2)
with colx1:
    st.subheader("📝 Journaliser la prévision")
    commentaire = st.text_area("Commentaires / hypothèses", placeholder="Renseigner les hypothèses clés…")
    btn_export = st.button("Exporter le PV (JSON)")

if btn_export:
    pv = {
        "operateur": {"nom": nom, "fonction": fonction, "departement": departement},
        "profil_tarif": tarif_nom,
        "prix_hp": prix_hp,
        "prix_hc": prix_hc,
        "horizon": f"{horizon} {horizon_unit}",
        "variables": {
            "temperature": temperature,
            "humidite": humidite,
            "precipitations": precipitations,
            "ensoleillement": ensoleillement,
            "population": population,
            "evenement": evenement,
            "prod_gwh": {
                "fossiles": prod_fossiles_gwh,
                "solaire": prod_solaire_gwh,
                "hydraulique": prod_hydraulique_gwh,
                "eolien": prod_eolien_gwh,
                "biomasse": prod_biomasse_gwh,
                "biogaz": prod_biogaz_gwh,
            },
        },
        "cap_mobilisables_mwh": {
            "batteries": cap_batterie_mwh,
            "thermique": cap_thermique_mwh,
            "effacement": effacement_mwh,
        },
        "resultats_resume": {
            "conso_tot_kwh": conso_tot,
            "prod_tot_kwh": prod_tot,
            "solde_kwh": delta_tot,
            "croisement": None if first_deficit_idx is None else str(res.iloc[first_deficit_idx]["timestamp"]),
        },
        "zones_top": zones.head(5).to_dict(orient="records"),
        "commentaire": commentaire,
        "horodatage": dt.datetime.now().isoformat(),
    }
    b = io.BytesIO()
    b.write(json.dumps(pv, ensure_ascii=False, indent=2).encode("utf-8"))
    b.seek(0)
    st.download_button("Télécharger le PV .json", b, file_name="PV_prevision_energie.json", mime="application/json")

# =============================
#        NOTES D’INTEGRATION
# =============================
st.markdown(
    """
**Intégration modèle (remplacement du MockModel)**  
- Charger votre **LSTM** (R²≈0.95) ou **XGBoost** entraîné (features conformes aux variables ci-dessus).  
- Remplacer `MockModel.predict_*` par vos appels `.predict()` (attention aux unités : GWh→kWh).  
- Si vous disposez d'une prédiction de production séparée (par source), agrégez-les ici avant l'affichage.  
- Pour la production temps réel, brancher l'ETL (EDF/Météo/INSEE/événements) et rafraîchir `features`.
"""
)

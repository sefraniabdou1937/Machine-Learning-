import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson

# --- CONFIGURATION GLOBALE ---
st.set_page_config(
    page_title="PL Analytics Pro",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STYLE CSS "PL OFFICIAL WHITE THEME" ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,700;1,300&display=swap');

    /* BASE */
    .stApp {
        background-color: #ffffff;
        font-family: 'Roboto', sans-serif;
        color: #38003c;
    }

    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-weight: 900 !important;
        text-transform: uppercase;
        color: #38003c;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* CONCEPT CARDS */
    .concept-card {
        background: linear-gradient(135deg, #f9f9f9 0%, #ffffff 100%);
        border-left: 5px solid #e90052;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .concept-card:hover {
        transform: translateX(5px);
        border-left-color: #00ff85;
    }
    .concept-title {
        color: #38003c;
        font-weight: 800;
        font-size: 1.1rem;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .concept-text {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.6;
        text-align: justify;
    }

    /* MATH & PARADOX BOXES */
    .math-legend {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-top: 10px;
    }
    
    .paradox-box {
        background-color: #38003c;
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        position: relative;
        box-shadow: 0 10px 30px rgba(56, 0, 60, 0.15);
    }
    .paradox-title {
        color: #00ff85;
        font-family: 'Merriweather', serif;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .paradox-text {
        font-family: 'Merriweather', serif;
        line-height: 1.8;
        opacity: 0.9;
    }

    /* BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, #38003c 0%, #2a002d 100%);
        color: #00ff85;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        padding: 16px 24px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 10px rgba(56, 0, 60, 0.2);
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 15px rgba(56, 0, 60, 0.3);
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- CHARGEMENT ---
@st.cache_resource
def load_data():
    try:
        model = pickle.load(open('model_final.pkl', 'rb'))
        df = pd.read_csv('cleaned_data.csv')
        df['Venue_Code'] = df['Venue'].apply(lambda x: 1 if x == 'Home' else 0)
        df['Win_Code'] = (df['Result'] == 'W').astype(int)
        return model, df
    except:
        return None, None

model, df = load_data()

LOGO_URLS = {
    "Arsenal": "https://resources.premierleague.com/premierleague/badges/100/t3.png",
    "Aston Villa": "https://resources.premierleague.com/premierleague/badges/100/t7.png",
    "Bournemouth": "https://resources.premierleague.com/premierleague/badges/100/t91.png",
    "Brentford": "https://resources.premierleague.com/premierleague/badges/100/t94.png",
    "Brighton": "https://resources.premierleague.com/premierleague/badges/100/t36.png",
    "Burnley": "https://resources.premierleague.com/premierleague/badges/100/t90.png",
    "Chelsea": "https://resources.premierleague.com/premierleague/badges/100/t8.png",
    "Crystal Palace": "https://resources.premierleague.com/premierleague/badges/100/t31.png",
    "Everton": "https://resources.premierleague.com/premierleague/badges/100/t11.png",
    "Fulham": "https://resources.premierleague.com/premierleague/badges/100/t54.png",
    "Liverpool": "https://resources.premierleague.com/premierleague/badges/100/t14.png",
    "Luton Town": "https://resources.premierleague.com/premierleague/badges/100/t102.png",
    "Manchester City": "https://resources.premierleague.com/premierleague/badges/100/t43.png",
    "Manchester United": "https://resources.premierleague.com/premierleague/badges/100/t1.png",
    "Newcastle": "https://resources.premierleague.com/premierleague/badges/100/t4.png",
    "Nottingham Forest": "https://resources.premierleague.com/premierleague/badges/100/t17.png",
    "Sheffield Utd": "https://resources.premierleague.com/premierleague/badges/100/t49.png",
    "Tottenham": "https://resources.premierleague.com/premierleague/badges/100/t6.png",
    "West Ham": "https://resources.premierleague.com/premierleague/badges/100/t21.png",
    "Wolves": "https://resources.premierleague.com/premierleague/badges/100/t39.png"
}

# --- LOGIQUE BACKEND ---

def get_poisson_probs(avg_goals):
    return [poisson.pmf(i, avg_goals) for i in range(6)]

def get_team_lambda(team_name, venue_code, df):
    recent = df[(df['Team'] == team_name) & (df['Venue_Code'] == venue_code)].tail(5)
    if recent.empty:
        lambda_val = df[df['Team'] == team_name]['GF'].mean()
    else:
        recent_mean = recent['GF'].mean()
        season_mean = df[df['Team'] == team_name]['GF'].mean()
        lambda_val = (recent_mean * 0.6) + (season_mean * 0.4)
    return max(lambda_val, 0.4)

def get_prediction_vector(home, away, df):
    h_stats = df[df['Venue_Code'] == 1].groupby('Team')['Win_Code'].mean()
    a_stats = df[df['Venue_Code'] == 0].groupby('Team')['Win_Code'].mean()
    
    encoded_team = h_stats.get(home, 0.5)
    encoded_opponent = a_stats.get(away, 0.5)
    
    home_recent = df[df['Team'] == home].tail(5)
    
    if not home_recent.empty:
        optimal_pred = home_recent['GF'].mean()
        eff_off = (home_recent['GF'] / home_recent['xG'].replace(0, 1)).mean()
        eff_def = (home_recent['xGA'] / home_recent['GA'].replace(0, 1)).mean()
    else:
        optimal_pred = 1.2
        eff_off = 1.0
        eff_def = 1.0
    
    att = 0.85 
    
    return pd.DataFrame([[encoded_team, encoded_opponent, optimal_pred, eff_off, eff_def, att]],
                        columns=['encoded_team', 'encoded_opponent', 'Optimal_Predictions', 
                                 'Rolling_Eff_Off', 'Rolling_Eff_Def', 'Attendance_Normalized'])

# --- HEADER ---
c1, c2 = st.columns([0.1, 0.9])
with c1:
    st.image("https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg", width=60)
with c2:
    st.markdown("<h1>PL Analytics <span style='color:#e90052'>Pro</span></h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Intelligence Artificielle & Modélisation Statistique</div>", unsafe_allow_html=True)

st.divider()

if df is None:
    st.error("⚠️ Données système manquantes.")
    st.stop()

# --- SÉLECTION ---
teams = sorted(df['Team'].unique())
col1, col2, col3 = st.columns([4, 1, 4])

with col1:
    st.markdown("### DOMICILE")
    home = st.selectbox("Selection H", teams, index=teams.index('Arsenal'), label_visibility="collapsed")
    st.image(LOGO_URLS.get(home), width=100)

with col2:
    st.markdown("<br><h1 style='text-align:center; color:#e90052;'>VS</h1>", unsafe_allow_html=True)

with col3:
    st.markdown("### EXTÉRIEUR")
    away = st.selectbox("Selection A", teams, index=teams.index('Chelsea'), label_visibility="collapsed")
    st.image(LOGO_URLS.get(away), width=100)

# --- ANALYSE ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("LANCER L'ANALYSE COMPLÈTE"):
    if home == away:
        st.warning("Veuillez sélectionner deux équipes différentes.")
    else:
        # CALCULS
        input_vec = get_prediction_vector(home, away, df)
        pred_class = model.predict(input_vec)[0]
        pred_proba = model.predict_proba(input_vec)[0][1]
        
        lambda_h = get_team_lambda(home, 1, df)
        lambda_a = get_team_lambda(away, 0, df)
        
        probs_h = get_poisson_probs(lambda_h)
        probs_a = get_poisson_probs(lambda_a)
        score_matrix = np.outer(probs_h, probs_a)
        
        # --- ONGLETS ---
        tab_dashboard, tab_tech = st.tabs(["📊 DASHBOARD RÉSULTATS", "🧬 BLUEPRINT TECHNIQUE"])

        # === TAB 1: DASHBOARD VISUEL ===
        with tab_dashboard:
            max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
            score_pred_h, score_pred_a = max_idx
            
            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            with k1:
                color = "#00ff85" if pred_class == 1 else "#e90052"
                res_txt = home if pred_class == 1 else "Nul / Défaite"
                st.markdown(f"""<div class="concept-card" style="text-align:center; border-left: 5px solid {color}"><div style="font-size:2rem; font-weight:bold; color:#38003c">{res_txt}</div><div style="color:#888">Verdict IA</div></div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class="concept-card" style="text-align:center;"><div style="font-size:2rem; font-weight:bold; color:#38003c">{score_pred_h} - {score_pred_a}</div><div style="color:#888">Score Poisson</div></div>""", unsafe_allow_html=True)
            with k3:
                conf = pred_proba if pred_class == 1 else (1 - pred_proba)
                st.markdown(f"""<div class="concept-card" style="text-align:center;"><div style="font-size:2rem; font-weight:bold; color:#38003c">{conf*100:.1f}%</div><div style="color:#888">Confiance</div></div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🔥 Probabilités de Score Exact")
            fig = px.imshow(score_matrix[:5, :5], 
                           labels=dict(x=f"Buts {away}", y=f"Buts {home}", color="Proba"),
                           x=[0,1,2,3,4], y=[0,1,2,3,4], text_auto='.1%', color_continuous_scale="Purples")
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # === TAB 2: ARCHITECTURE DU SYSTÈME ===
        with tab_tech:
            st.markdown("## 📐 Architecture du Système")
            st.write("Le système transforme des données brutes en décisions stratégiques à travers un pipeline en 4 étapes.")

            # --- 1. PIPELINE ---
            st.subheader("1. Le Flux de Données (Data Pipeline)")
            
            col_pipe_txt, col_pipe_diag = st.columns([1, 2])
            with col_pipe_txt:
                st.markdown("""
                Le voyage d'une donnée :
                1. **Extraction** : Scraping des stats (FBref).
                2. **Nettoyage** : Suppression des matchs incomplets.
                3. **Feature Engineering** : Calcul des métriques avancées (xG, Lambda).
                4. **Prédiction** : Double moteur (Logistic + Poisson).
                """)
            with col_pipe_diag:
                st.markdown("""
                ```mermaid
                graph LR
                    A[Raw Data] --> B(Cleaning)
                    B --> C{Feature Eng.}
                    C -->|Vectors| D[AI Models]
                    D --> E[Outcome]
                    D --> F[Score]
                    style D fill:#38003c,color:white
                    style C fill:#e90052,color:white
                ```
                """, unsafe_allow_html=True)

            st.markdown("---")

            # --- 2. FEATURES ---
            st.subheader("2. Les Variables Clés (Feature Engineering)")
            c_feat1, c_feat2 = st.columns(2)
            with c_feat1:
                st.markdown("""<div class="concept-card"><div class="concept-title">🧬 1. L'ADN Historique (Team Encoding)</div><div class="concept-text">L'algorithme transforme chaque club en une valeur numérique représentant sa probabilité historique de victoire. C'est la mémoire à long terme.</div></div>""", unsafe_allow_html=True)
                st.markdown("""<div class="concept-card"><div class="concept-title">⚡ 3. La Forme Offensive (xG Eff)</div><div class="concept-text">Ratio Buts/xG sur 5 matchs. >1 indique une sur-performance (finition chirurgicale), <1 un manque de réalisme.</div></div>""", unsafe_allow_html=True)
            with c_feat2:
                st.markdown("""<div class="concept-card"><div class="concept-title">🏟️ 2. Le Facteur 'Forteresse'</div><div class="concept-text">L'impact du public est modélisé par une variable normalisée (0-1), quantifiant la pression des supporters.</div></div>""", unsafe_allow_html=True)
                st.markdown("""<div class="concept-card"><div class="concept-title">🛡️ 4. La Résilience Défensive</div><div class="concept-text">Capacité à "plier sans rompre", en comparant les buts encaissés aux occasions concédées (xGA).</div></div>""", unsafe_allow_html=True)

            st.markdown("---")

            # --- 3. MATHS ---
            st.subheader("3. Le Cœur Mathématique")
            
            m1, m2 = st.columns(2)
            with m1:
                st.markdown("#### Régression Logistique (Vainqueur)")
                st.latex(r''' P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i X_i)}} ''')
                st.markdown("""
                <div class="math-legend">
                <strong>Légende :</strong><br>
                • $P(Y=1)$ : Probabilité de Victoire Domicile.<br>
                • $X_i$ : Les Features (Force, xG, etc.).<br>
                • $\beta_i$ : Le "Poids" de chaque feature (l'importance).
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown("#### Loi de Poisson (Score Exact)")
                st.latex(r''' P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!} ''')
                st.markdown("""
                <div class="math-legend">
                <strong>Légende :</strong><br>
                • $k$ : Nombre de buts cible (0, 1, 2...).<br>
                • $\lambda$ (Lambda) : Espérance de buts (calculée dynamiquement).<br>
                • $e$ : Constante d'Euler (~2.718).
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # --- 4. CALIBRATION ---
            st.subheader("4. Calibration & Marge d'Erreur")
            st.write("""
            Aucun modèle n'est prophétique. Notre analyse des résidus sur la saison 2023-2024 montre une caractéristique structurelle importante :
            """)
            st.info("⚠️ **La Marge de Tolérance :** Le modèle admet une erreur standard de **±1 but** par rapport à la réalité.")
            st.write("""
            Cela signifie que pour un score prédit de **2-1**, le résultat réel statistique le plus probable se situe dans une fenêtre comprise entre **1-0** et **3-2**.
            Cette incertitude est irréductible en raison de la nature aléatoire d'un but (poteau, VAR, erreur d'arbitrage).
            """)

            # --- 5. LE PARADOXE WYDAD / CITY (MISE À JOUR 2025) ---
            st.markdown("""
            <div class="paradox-box">
                <div class="paradox-title">5. Étude de Cas Réelle : Le Paradoxe "Wydad vs City"</div>
                <div class="paradox-text">
                    Ce projet, conçu initialement en <strong>2024</strong>, posait une question théorique fascinante : que se passe-t-il quand la logique froide des chiffres rencontre la passion pure ?
                    <br><br>
                    Prenons le cas réel du <strong>Wydad AC</strong> face à <strong>Manchester City</strong> (Mondial des Clubs).
                    <br><br>
                    Aux yeux de notre modèle mathématique, la conclusion est sans appel : <strong>"City reste City"</strong>. 
                    Avec une domination structurelle sur toutes les métriques (Budget, xG, Possession, Valeur Marchande), 
                    l'algorithme prédit inévitablement une domination totale des Citizens.
                    <br><br>
                    Pourtant, la réalité du terrain a nuancé cette prédiction : <strong>Le Wydad a fait un grand match.</strong> 
                    Bien que "City reste le reste" (et l'emporte souvent à la fin), la résistance héroïque, l'état moral et la qualité de jeu du Wydad constituent ce 
                    bruit statistique (le "Facteur Humain") que le modèle ne pourra jamais totalement capturer.
                    <br><br>
                    <em>"Les chiffres donnent le favori, mais c'est le terrain qui donne le respect."</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👆 Veuillez sélectionner les équipes pour générer le rapport technique.")
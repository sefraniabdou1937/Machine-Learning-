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



# --- STYLE CSS "PL LIGHT THEME" ---

st.markdown("""

    <style>

    /* Fond global Blanc */

    .stApp {

        background-color: #ffffff;

        color: #38003c; /* PL Dark Purple */

    }

    

    /* Titres en Violet PL */

    h1, h2, h3, h4, h5 {

        font-family: 'Roboto', sans-serif;

        font-weight: 900;

        color: #38003c;

    }

    

    /* Bouton Principal - Style "Action" */

    .stButton>button {

        background-color: #38003c;

        color: #00ff85; /* PL Neon Green */

        font-weight: bold;

        border-radius: 8px;

        height: 3.5em;

        width: 100%;

        border: 2px solid #38003c;

        transition: all 0.3s;

        text-transform: uppercase;

        letter-spacing: 1px;

    }

    .stButton>button:hover {

        background-color: #ffffff;

        color: #38003c;

        border-color: #00ff85;

    }



    /* Cards (Conteneurs Blancs avec Ombres) */

    .metric-card {

        background-color: #f4f4f4;

        padding: 20px;

        border-radius: 12px;

        border-left: 5px solid #e90052; /* PL Pink */

        box-shadow: 0 2px 5px rgba(0,0,0,0.05);

        text-align: center;

    }

    

    /* Métriques */

    .big-number {

        font-size: 2.2rem;

        font-weight: 800;

        color: #38003c;

    }

    .small-label {

        font-size: 0.85rem;

        color: #666;

        text-transform: uppercase;

        font-weight: 600;

    }

    

    /* Séparateurs personnalisés */

    hr {

        border-top: 2px solid #00ff85;

        margin-top: 2rem;

        margin-bottom: 2rem;

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

    "Arsenal": "https://resources.premierleague.com/premierleague/badges/50/t3.png",

    "Aston Villa": "https://resources.premierleague.com/premierleague/badges/50/t7.png",

    "Bournemouth": "https://resources.premierleague.com/premierleague/badges/50/t91.png",

    "Brentford": "https://resources.premierleague.com/premierleague/badges/50/t94.png",

    "Brighton": "https://resources.premierleague.com/premierleague/badges/50/t36.png",

    "Burnley": "https://resources.premierleague.com/premierleague/badges/50/t90.png",

    "Chelsea": "https://resources.premierleague.com/premierleague/badges/50/t8.png",

    "Crystal Palace": "https://resources.premierleague.com/premierleague/badges/50/t31.png",

    "Everton": "https://resources.premierleague.com/premierleague/badges/50/t11.png",

    "Fulham": "https://resources.premierleague.com/premierleague/badges/50/t54.png",

    "Liverpool": "https://resources.premierleague.com/premierleague/badges/50/t14.png",

    "Luton Town": "https://resources.premierleague.com/premierleague/badges/50/t102.png",

    "Manchester City": "https://resources.premierleague.com/premierleague/badges/50/t43.png",

    "Manchester United": "https://resources.premierleague.com/premierleague/badges/50/t1.png",

    "Newcastle": "https://resources.premierleague.com/premierleague/badges/50/t4.png",

    "Nottingham Forest": "https://resources.premierleague.com/premierleague/badges/50/t17.png",

    "Sheffield Utd": "https://resources.premierleague.com/premierleague/badges/50/t49.png",

    "Tottenham": "https://resources.premierleague.com/premierleague/badges/50/t6.png",

    "West Ham": "https://resources.premierleague.com/premierleague/badges/50/t21.png",

    "Wolves": "https://resources.premierleague.com/premierleague/badges/50/t39.png"

}



# --- FONCTIONS ANALYTIQUES ---



def get_poisson_probs(avg_goals):

    """Calcule la loi de Poisson pour 0 à 5 buts"""

    probs = [poisson.pmf(i, avg_goals) for i in range(6)]

    return probs



def get_team_lambda(team_name, venue_code, df):

    """Estime le paramètre Lambda (Moyenne de buts attendus)"""

    # Moyenne des 5 derniers matchs à domicile/extérieur

    recent_form = df[(df['Team'] == team_name) & (df['Venue_Code'] == venue_code)].tail(5)

    if recent_form.empty:

        lambda_val = df[df['Team'] == team_name]['GF'].mean() # Fallback global

    else:

        lambda_val = recent_form['GF'].mean()

    return max(lambda_val, 0.5) # Minimum 0.5 but pour éviter lambda=0



def get_prediction_vector(home, away, df):

    # Logique identique pour nourrir le modèle LogisticRegression

    h_stats = df[df['Venue_Code'] == 1].groupby('Team')['Win_Code'].mean()

    a_stats = df[df['Venue_Code'] == 0].groupby('Team')['Win_Code'].mean()

    

    encoded_team = h_stats.get(home, 0.5)

    encoded_opponent = a_stats.get(away, 0.5)

    

    # Feature Engineering (Simplifié pour l'app)

    home_recent = df[df['Team'] == home].tail(5)

    optimal_pred = home_recent['GF'].mean() if not home_recent.empty else 1.3

    

    eff_off = (home_recent['GF'] / home_recent['xG'].replace(0, 1)).mean() if not home_recent.empty else 1.0

    eff_def = (home_recent['xGA'] / home_recent['GA'].replace(0, 1)).mean() if not home_recent.empty else 1.0

    

    att_norm = 0.8 # Valeur par défaut haute pour un match important

    

    return pd.DataFrame([[encoded_team, encoded_opponent, optimal_pred, eff_off, eff_def, att_norm]],

                       columns=['encoded_team', 'encoded_opponent', 'Optimal_Predictions', 

                                'Rolling_Eff_Off', 'Rolling_Eff_Def', 'Attendance_Normalized'])



# --- HEADER ---

col_logo, col_title = st.columns([1, 8])

with col_logo:

    st.image("https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg", width=80)

with col_title:

    st.markdown("# PREMIER LEAGUE ANALYTICS HUB")

    st.caption("Intelligence Artificielle & Modélisation Statistique")



# --- SÉLECTION DES ÉQUIPES ---

if df is not None:

    teams = sorted(df['Team'].unique())

    

    # Zone de contrôle stylisée

    with st.container():

        c1, c2, c3 = st.columns([3, 1, 3])

        with c1:

            home = st.selectbox("DOMICILE", teams, index=teams.index('Liverpool'))

            st.image(LOGO_URLS.get(home, ""), width=100)

        with c2:

            st.markdown("<h1 style='text-align:center; padding-top: 30px; color:#e90052;'>VS</h1>", unsafe_allow_html=True)

        with c3:

            away = st.selectbox("EXTÉRIEUR", teams, index=teams.index('Manchester City'))

            st.image(LOGO_URLS.get(away, ""), width=100)



    # --- ANALYSE ---

    if st.button("LANCER L'ANALYSE DU MATCH"):

        if home == away:

            st.error("Les équipes doivent être différentes.")

        else:

            # 1. Prédictions du Modèle (Machine Learning)

            input_vec = get_prediction_vector(home, away, df)

            pred_class = model.predict(input_vec)[0]

            pred_proba = model.predict_proba(input_vec)[0][1]



            # 2. Prédictions Statistiques (Loi de Poisson)

            lambda_home = get_team_lambda(home, 1, df)

            lambda_away = get_team_lambda(away, 0, df)

            

            # Calcul des probabilités de scores exacts (Matrice)

            # P(Score = i-j) = P(Home=i) * P(Away=j)

            probs_home = get_poisson_probs(lambda_home)

            probs_away = get_poisson_probs(lambda_away)

            

            # Score le plus probable (Index du max)

            most_likely_home = np.argmax(probs_home)

            most_likely_away = np.argmax(probs_away)



            # --- AFFICHAGE DES RÉSULTATS ---

            

            # BLOC 1 : VERDICT IA

            st.markdown("---")

            st.markdown("### 🤖 VERDICT DE L'INTELLIGENCE ARTIFICIELLE")

            

            res1, res2, res3 = st.columns(3)

            with res1:

                st.markdown(f"""

                <div class="metric-card">

                    <div class="small-label">Vainqueur Prédit</div>

                    <div class="big-number" style="color: {'#00ff85' if pred_class==1 else '#e90052'};">

                        {home if pred_class == 1 else "Non-Victoire"}

                    </div>

                </div>

                """, unsafe_allow_html=True)

            

            with res2:

                st.markdown(f"""

                <div class="metric-card">

                    <div class="small-label">Confiance Modèle</div>

                    <div class="big-number">{pred_proba*100:.1f}%</div>

                </div>

                """, unsafe_allow_html=True)

                

            with res3:

                st.markdown(f"""

                <div class="metric-card">

                    <div class="small-label">Buts Attendus (Home)</div>

                    <div class="big-number">{lambda_home:.2f}</div>

                </div>

                """, unsafe_allow_html=True)



            # BLOC 2 : ANALYSE POISSON (Le détail demandé)

            st.markdown("---")

            st.markdown(f"### 📊 LOI DE POISSON : SCÉNARIOS DE BUTS")

            st.write("Ce graphique montre la probabilité (en %) que chaque équipe marque un nombre précis de buts, basée sur leur performance offensive récente.")



            # Création du graphique Poisson (Bar Chart Groupé)

            goals_range = list(range(6))

            

            fig_poisson = go.Figure()

            fig_poisson.add_trace(go.Bar(

                x=goals_range, 

                y=[p*100 for p in probs_home], 

                name=home,

                marker_color='#38003c' # Violet PL

            ))

            fig_poisson.add_trace(go.Bar(

                x=goals_range, 

                y=[p*100 for p in probs_away], 

                name=away,

                marker_color='#00ff85' # Vert PL

            ))

            

            fig_poisson.update_layout(

                title="Distribution de Probabilité des Buts",

                xaxis_title="Nombre de Buts",

                yaxis_title="Probabilité (%)",

                barmode='group',

                plot_bgcolor='rgba(0,0,0,0)',

                paper_bgcolor='rgba(0,0,0,0)',

                font=dict(color='#38003c')

            )

            st.plotly_chart(fig_poisson, use_container_width=True)



            # Explication pédagogique "En Clair"

            with st.expander("ℹ️ Comprendre ce graphique (Loi de Poisson)"):

                st.write(f"""

                La loi de Poisson est utilisée pour prédire des événements rares (comme les buts).

                - **Lambda ({home})** = {lambda_home:.2f} buts/match en moyenne.

                - **Lambda ({away})** = {lambda_away:.2f} buts/match en moyenne.

                

                La barre la plus haute pour chaque équipe indique le nombre de buts qu'ils ont le plus de chances de marquer aujourd'hui.

                """)



            # BLOC 3 : LE SCORE EXACT (Le "Optimal Goal Prediction")

            st.markdown("---")

            col_score1, col_score2 = st.columns([1, 1])

            

            with col_score1:

                st.markdown("### 🎯 SCORE LE PLUS PROBABLE")

                st.markdown(f"""

                <div style="background-color: #38003c; color: white; padding: 30px; border-radius: 15px; text-align: center;">

                    <span style="font-size: 4rem; font-weight: bold;">{most_likely_home} - {most_likely_away}</span>

                </div>

                """, unsafe_allow_html=True)

            

            with col_score2:

                st.markdown("### ⚡ EFFICACITÉ OFFENSIVE")

                # Comparaison des indices d'efficacité (Rolling Eff)

                eff_h = input_vec['Rolling_Eff_Off'][0]

                

                # Jauge simple

                fig_bullet = go.Figure(go.Indicator(

                    mode = "number+gauge+delta",

                    value = eff_h,

                    title = {"text": f"Efficacité {home} vs Moyenne Ligue"},

                    delta = {'reference': 1.0},

                    gauge = {

                        'shape': "bullet",

                        'axis': {'range': [0, 2]},

                        'bar': {'color': "#e90052"},

                        'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 1.0}

                    }

                ))

                fig_bullet.update_layout(height=200, margin={'t':0, 'b':0})

                st.plotly_chart(fig_bullet, use_container_width=True)



else:

    st.warning("Veuillez charger les fichiers de données (model_final.pkl & cleaned_data.csv).")
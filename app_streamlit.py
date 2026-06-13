import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('modelo_goles.pkl')

def predict_match(elo_a, elo_b, odd_a, odd_draw, odd_b):
    def _build_row(home_elo, away_elo, odd_home, odd_away):
        prob_h = 1 / odd_home
        prob_d = 1 / odd_draw
        prob_a = 1 / odd_away
        total  = prob_h + prob_d + prob_a
        return pd.DataFrame([{
            'EloDiff':      home_elo - away_elo,
            'EloRatio':     home_elo / away_elo,
            'ExpGoalsHome': 1.5 * (home_elo / away_elo) ** 0.5,
            'ExpGoalsAway': 1.2 * (away_elo / home_elo) ** 0.5,
            'ProbHomeNorm': prob_h / total,
            'ProbDrawNorm': prob_d / total,
            'ProbAwayNorm': prob_a / total,
            'OddHome': odd_home, 'OddDraw': odd_draw, 'OddAway': odd_away,
        }])

    pred1 = np.expm1(model.predict(_build_row(elo_a, elo_b, odd_a, odd_b))[0])
    pred2 = np.expm1(model.predict(_build_row(elo_b, elo_a, odd_b, odd_a))[0])

    goals_a = float(np.clip((pred1[0] + pred2[1]) / 2, 0, None))
    goals_b = float(np.clip((pred1[1] + pred2[0]) / 2, 0, None))
    return round(goals_a, 2), round(goals_b, 2)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚽ Predictor de Goles")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Equipo A")
    elo_a = st.number_input("Elo", value=1686, key="elo_a")
with col2:
    st.subheader("Equipo B")
    elo_b = st.number_input("Elo", value=1586, key="elo_b")

st.divider()
st.subheader("Cuotas")
col3, col4, col5 = st.columns(3)
with col3: odd_a    = st.number_input("Gana A",  value=1.65, step=0.01)
with col4: odd_draw = st.number_input("Empate",  value=3.30, step=0.01)
with col5: odd_b    = st.number_input("Gana B",  value=4.30, step=0.01)

st.divider()

if st.button("Predecir", use_container_width=True, type="primary"):
    goals_a, goals_b = predict_match(elo_a, elo_b, odd_a, odd_draw, odd_b)

    col6, col7, col8 = st.columns([2, 1, 2])
    with col6:
        st.metric("Equipo A", f"{goals_a} goles")
    with col7:
        st.markdown("<h3 style='text-align:center;margin-top:20px'>VS</h3>", unsafe_allow_html=True)
    with col8:
        st.metric("Equipo B", f"{goals_b} goles")

    if abs(goals_a - goals_b) < 0.15:
        st.info("Partido muy parejo — empate probable")
    elif goals_a > goals_b:
        st.success(f"Equipo A favorito por {round(goals_a - goals_b, 2)} goles")
    else:
        st.success(f"Equipo B favorito por {round(goals_b - goals_a, 2)} goles")
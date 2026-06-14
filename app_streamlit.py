import streamlit as st
import joblib
import numpy as np
import pandas as pd

model_goals  = joblib.load('modelo_goles.pkl')
model_result = joblib.load('modelo_resultado.pkl')


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

    def _build_row_ext(home_elo, away_elo, odd_home, odd_away, pred_home, pred_away):
        row = _build_row(home_elo, away_elo, odd_home, odd_away)
        row['PredGoalsHome'] = pred_home
        row['PredGoalsAway'] = pred_away
        return row

    # Goles primero
    pred1   = np.expm1(model_goals.predict(_build_row(elo_a, elo_b, odd_a, odd_b))[0])
    pred2   = np.expm1(model_goals.predict(_build_row(elo_b, elo_a, odd_b, odd_a))[0])
    goals_a = float(np.clip((pred1[0] + pred2[1]) / 2, 0, None))
    goals_b = float(np.clip((pred1[1] + pred2[0]) / 2, 0, None))

    # Clasificador con goles incluidos en el row
    proba1  = model_result.predict_proba(_build_row_ext(elo_a, elo_b, odd_a, odd_b, goals_a, goals_b))[0]
    proba2  = model_result.predict_proba(_build_row_ext(elo_b, elo_a, odd_b, odd_a, goals_b, goals_a))[0]

    classes = model_result.classes_
    idx_H   = list(classes).index('H')
    idx_D   = list(classes).index('D')
    idx_A   = list(classes).index('A')

    prob_A = (proba1[idx_H] + proba2[idx_A]) / 2
    prob_D = (proba1[idx_D] + proba2[idx_D]) / 2
    prob_B = (proba1[idx_A] + proba2[idx_H]) / 2

    resultado  = {prob_A: 'Gana A', prob_D: 'Empate', prob_B: 'Gana B'}
    pred_label = resultado[max(prob_A, prob_D, prob_B)]

    return {
        'goals_a':   round(goals_a, 2),
        'goals_b':   round(goals_b, 2),
        'resultado': pred_label,
        'prob_a':    round(float(prob_A), 3),
        'prob_draw': round(float(prob_D), 3),
        'prob_b':    round(float(prob_B), 3),
    }

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚽ Predictor de Partidos")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Equipo A")
    elo_a = st.number_input("Elo", value=1686, step=1, key="elo_a")
with col2:
    st.subheader("Equipo B")
    elo_b = st.number_input("Elo", value=1586, step=1, key="elo_b")

st.divider()
st.subheader("Cuotas")
col3, col4, col5 = st.columns(3)
with col3: odd_a    = st.number_input("Gana A",  value=1.65, step=0.01)
with col4: odd_draw = st.number_input("Empate",  value=3.30, step=0.01)
with col5: odd_b    = st.number_input("Gana B",  value=4.30, step=0.01)

st.divider()

if st.button("Predecir", use_container_width=True, type="primary"):
    res = predict_match(elo_a, elo_b, odd_a, odd_draw, odd_b)

    # ── Goles ────────────────────────────────────────────────────────────────
    st.subheader("Goles estimados")
    col6, col7, col8 = st.columns([2, 1, 2])
    with col6:
        st.metric("Equipo A", f"{res['goals_a']}")
    with col7:
        st.markdown("<h3 style='text-align:center;margin-top:20px'>VS</h3>", unsafe_allow_html=True)
    with col8:
        st.metric("Equipo B", f"{res['goals_b']}")

    # ── Resultado ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Resultado esperado")

    color = {"Gana A": "🟢", "Empate": "🟡", "Gana B": "🔵"}
    st.markdown(f"### {color[res['resultado']]} {res['resultado']}")

    # ── Probabilidades ────────────────────────────────────────────────────────
    st.subheader("Probabilidades")
    col9, col10, col11 = st.columns(3)
    with col9:
        st.metric("Gana A",  f"{res['prob_a']:.1%}")
        st.progress(res['prob_a'])
    with col10:
        st.metric("Empate",  f"{res['prob_draw']:.1%}")
        st.progress(res['prob_draw'])
    with col11:
        st.metric("Gana B",  f"{res['prob_b']:.1%}")
        st.progress(res['prob_b'])

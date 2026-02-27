# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Preditor de Obesidade",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helper functions ----------
@st.cache_data
def load_data(path="obesity_tratado.csv"):
    return pd.read_csv(path)

@st.cache_data
def load_model(path="model_pipeline.joblib"):
    return joblib.load(path)

def calc_imc(weight, height):
    try:
        return weight / (height**2)
    except Exception:
        return np.nan

# ---------- Load resources (data + model) ----------
st.sidebar.title("Carregamento")
data_load_state = st.sidebar.text("Carregando dados...")
try:
    df = load_data("obesity_tratado.csv")
    data_load_state.text("Dados carregados ✅")
except Exception as e:
    data_load_state.text("Erro ao carregar dados. Coloque o CSV na pasta.")
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

model_load_state = st.sidebar.text("Carregando modelo...")
try:
    model = load_model("model_pipeline.joblib")
    model_load_state.text("Modelo carregado ✅")
except Exception as e:
    model = None
    model_load_state.text("Sem modelo (coloque model_pipeline.joblib na pasta)")
    st.warning("Modelo não encontrado. Ainda assim é possível explorar os dados.")
    # We don't stop; allow exploration

# ---------- Top header ----------
st.markdown(
    """
<div style="background-color:#1E90FF;padding:25px;border-radius:8px">
  <h1 style="color:white;margin:0">Análise & Previsor de Obesidade</h1>
  <p style="color:whitesmoke;margin:0">Interativo — explore os fatores e faça previsões com o modelo.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Summary metrics ----------
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    st.metric("Média IMC (amostra)", f"{df['IMC'].mean():.2f}")
with col2:
    st.metric("Total registros", f"{len(df):,}")
with col3:
    # exemplo: Contar categoria de obesidade grave (ajuste para o seu label)
    severe_label = "Obesity_Type_III"  # ajuste conforme seu dicionário
    count_severe = int((df["Obesity"] == severe_label).sum()) if "Obesity" in df.columns else 0
    st.metric("Obesidade Grave (ex.)", f"{count_severe:,}")
with col4:
    st.metric("Fem/Masc (sample)", f"F:{(df['Gender']=='Female').sum()} / M:{(df['Gender']=='Male').sum()}" if "Gender" in df.columns else "—")

st.write("---")

# ---------- Sidebar - filtros e inputs ----------
st.sidebar.header("Filtros e Previsão")
# Data filters
gender_filter = st.sidebar.multiselect("Gender", options=df["Gender"].unique() if "Gender" in df.columns else [], default=df["Gender"].unique().tolist() if "Gender" in df.columns else [])
obesity_filter = st.sidebar.multiselect("Obesity (categorias)", options=df["Obesity"].unique() if "Obesity" in df.columns else [], default=df["Obesity"].unique().tolist() if "Obesity" in df.columns else [])

st.sidebar.markdown("---")
st.sidebar.subheader("Fazer uma previsão (exemplo)")
# Inputs for prediction
pred_gender = st.sidebar.selectbox("Gênero", options=["Female","Male"])
pred_age = st.sidebar.number_input("Idade", min_value=10, max_value=100, value=30)
pred_height = st.sidebar.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
pred_weight = st.sidebar.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
pred_family = st.sidebar.selectbox("Histórico familiar?", options=["yes","no"])
pred_FAVC = st.sidebar.selectbox("Consome alimentos calóricos?", options=["yes","no"])
pred_FCVC = st.sidebar.selectbox("Frequência vegetais", options=[1,2,3])
pred_NCP = st.sidebar.selectbox("Refeições principais", options=[1,2,3,4])
pred_CAEC = st.sidebar.selectbox("Come entre refeições?", options=["no","Sometimes","Frequently","Always"])
pred_SMOKE = st.sidebar.selectbox("Fuma?", options=["yes","no"])
pred_CH2O = st.sidebar.selectbox("Consumo de água", options=[1,2,3])
pred_SCC = st.sidebar.selectbox("Monitora calorias?", options=["yes","no"])
pred_FAF = st.sidebar.selectbox("Atividade física (FAF)", options=[0,1,2,3])
pred_TUE = st.sidebar.selectbox("Tempo com eletrônicos (TUE)", options=[0,1,2])
pred_CALC = st.sidebar.selectbox("Consumo álcool", options=["no","Sometimes","Frequently","Always"])
pred_MTRANS = st.sidebar.selectbox("Transporte", options=df["MTRANS"].unique().tolist() if "MTRANS" in df.columns else ["Automobile","Public_Transport","Walking","Motorbike","Bike"])

st.sidebar.markdown("---")
download_btn = st.sidebar.button("Baixar dataset tratado (.csv)")

# ---------- Main page: charts and table ----------
left_col, right_col = st.columns([2,1])

with left_col:
    st.subheader("Distribuição por Classe de Obesidade")
    if "Obesity" in df.columns:
        obs_counts = df["Obesity"].value_counts().reset_index()
        obs_counts.columns = ["Obesity", "count"]
        fig1 = px.bar(obs_counts, x="count", y="Obesity", orientation="h", text="count")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Campo 'Obesity' não encontrado no dataset.")

    st.subheader("IMC - Distribuição")
    if "IMC" in df.columns:
        fig2 = px.histogram(df, x="IMC", nbins=30, marginal="box")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Campo 'IMC' não encontrado no dataset.")

    st.subheader("Atividade Física (FAF) x Obesidade")
    if "FAF" in df.columns and "Obesity" in df.columns:
        ct = df.groupby(["FAF","Obesity"]).size().reset_index(name="count")
        fig3 = px.bar(ct, x="FAF", y="count", color="Obesity", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Campos FAF / Obesity não disponíveis para este gráfico.")

with right_col:
    st.subheader("Amostra de dados")
    st.dataframe(df.head(200))

    st.subheader("Controles rápidos")
    st.write("Use os filtros na barra lateral para ajustar os gráficos.")
    if download_btn:
        st.success("Gerando arquivo para download...")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Clique para baixar CSV tratado", data=csv, file_name="obesity_tratado.csv", mime="text/csv")

# ---------- Prediction logic ----------
st.write("---")
st.subheader("Previsão com o modelo")

if model is None:
    st.warning("Modelo não carregado — coloque model_pipeline.joblib na pasta para habilitar previsões.")
else:
    input_df = pd.DataFrame([{
        "Gender": pred_gender,
        "Age": pred_age,
        "Height": pred_height,
        "Weight": pred_weight,
        "family_history": pred_family,
        "FAVC": pred_FAVC,
        "FCVC": pred_FCVC,
        "NCP": pred_NCP,
        "CAEC": pred_CAEC,
        "SMOKE": pred_SMOKE,
        "CH2O": pred_CH2O,
        "SCC": pred_SCC,
        "FAF": pred_FAF,
        "TUE": pred_TUE,
        "CALC": pred_CALC,
        "MTRANS": pred_MTRANS
    }])

    st.write("Dados de entrada:")
    st.table(input_df.T)

    if st.button("Prever com o modelo"):
        try:
            pred = model.predict(input_df)
            # tentar probabilidade se houver
            proba = None
            try:
                proba = model.predict_proba(input_df)
            except Exception:
                proba = None

            st.success(f"Classe prevista: {pred[0]}")
            if proba is not None:
                # mostrar probabilidades em tabela
                classes = model.classes_ if hasattr(model, "classes_") else None
                if classes is not None:
                    prob_df = pd.DataFrame(proba, columns=classes)
                    st.write("Probabilidades:")
                    st.dataframe(prob_df.T)
                else:
                    st.write("Probabilidades calculadas (labels desconhecidos):")
                    st.write(proba)
        except Exception as e:
            st.error(f"Erro durante a previsão: {e}")

# ---------- Footer ----------
st.write("---")
st.caption("App desenvolvido com Streamlit — ajuste os textos, cores e imagens conforme necessário.")
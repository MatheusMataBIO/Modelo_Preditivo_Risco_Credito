# ============================================================
# app.py — Previsão de Inadimplência | Datarisk
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
from datetime import date

# ── Configuração da página ────────────────────────────────────
st.set_page_config(
    page_title="Previsão de Inadimplência | Datarisk",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Carregar artefatos ────────────────────────────────────────
@st.cache_resource
def carregar_artefatos():
    base_path = os.path.dirname(os.path.abspath(__file__))
    modelo    = joblib.load(os.path.join(base_path, "artefatos", "modelo_final.pkl"))
    encoders  = joblib.load(os.path.join(base_path, "artefatos", "encoders.pkl"))
    params    = joblib.load(os.path.join(base_path, "artefatos", "params_imputacao.pkl"))
    validacao = joblib.load(os.path.join(base_path, "artefatos", "dados_validacao.pkl"))
    return modelo, encoders, params, validacao

# Expor no escopo global
modelo, encoders, params, validacao = carregar_artefatos()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image(
    "https://img.shields.io/badge/LightGBM-Calibrado-3C3489",
    width=200
)
st.sidebar.markdown("## Navegação")
pagina = st.sidebar.radio(
    "",
    ["💳 Previsão Individual",
     "📊 Performance do Modelo",
     "ℹ️ Sobre o Projeto"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Modelo:** LightGBM + Isotonic Calibration

**AUC-ROC:** 0.9444

**Gini:** 0.8888

**KS:** 0.7573

**Brier:** 0.0362
""")

# ── Funções auxiliares ────────────────────────────────────────
FAIXAS = [
    (0.05,  "🟢 MUITO BAIXO",  "#16A34A", "Monitoramento padrão",
     "Nenhuma ação necessária.", "Baixa"),
    (0.10,  "🟡 BAIXO",        "#CA8A04", "Acompanhamento leve",
     "Envio de lembrete automático próximo ao vencimento.", "Baixa"),
    (0.20,  "🟠 MODERADO",     "#EA580C", "Acompanhamento ativo",
     "Lembrete antecipado 5 dias antes do vencimento.", "Média"),
    (0.30,  "🔴 ALTO",         "#DC2626", "Contato proativo",
     "Ligação ou WhatsApp antes do vencimento.", "Alta"),
    (0.50,  "🔴 MUITO ALTO",   "#B91C1C", "Ação imediata",
     "Contato direto com gestor. Avaliar renegociação.", "Máxima"),
    (1.01,  "🚨 CRÍTICO",      "#7F1D1D", "Risco severo",
     "Acionar equipe especializada. Considerar garantias adicionais.",
     "Urgente"),
]

def classificar_risco(prob):
    for limite, faixa, cor, acao, detalhe, prioridade in FAIXAS:
        if prob < limite:
            return faixa, cor, acao, detalhe, prioridade
    return FAIXAS[-1][1:]

mapa_ddd = {
    "11":"SP","12":"SP","13":"SP","14":"SP","15":"SP",
    "16":"SP","17":"SP","18":"SP","19":"SP",
    "21":"RJ","22":"RJ","24":"RJ","27":"ES","28":"ES",
    "31":"MG","32":"MG","33":"MG","34":"MG","35":"MG",
    "37":"MG","38":"MG","41":"PR","42":"PR","43":"PR",
    "44":"PR","45":"PR","46":"PR","47":"SC","48":"SC",
    "49":"SC","51":"RS","53":"RS","54":"RS","55":"RS",
    "61":"DF","62":"GO","63":"TO","64":"GO","65":"MT",
    "66":"MT","67":"MS","68":"AC","69":"RO","71":"BA",
    "73":"BA","74":"BA","75":"BA","77":"BA","79":"SE",
    "81":"PE","82":"AL","83":"PB","84":"RN","85":"CE",
    "86":"PI","87":"PE","88":"CE","89":"PI","91":"PA",
    "92":"AM","93":"PA","94":"PA","95":"RR","96":"AP",
    "97":"AM","98":"MA","99":"MA",
}

def cep_para_regiao(cep):
    try:
        cep = int(cep)
    except:
        return "DESCONHECIDO"
    if   1  <= cep <=  9: return "SP_Capital"
    elif 10 <= cep <= 19: return "SP_Interior"
    elif 20 <= cep <= 28: return "RJ"
    elif cep == 29:       return "ES"
    elif 30 <= cep <= 39: return "MG"
    elif 40 <= cep <= 48: return "BA"
    elif cep == 49:       return "SE"
    elif 50 <= cep <= 56: return "PE"
    elif cep == 57:       return "AL"
    elif cep == 58:       return "PB"
    elif cep == 59:       return "RN"
    elif 60 <= cep <= 63: return "CE"
    elif cep == 64:       return "PI"
    elif cep == 65:       return "MA"
    elif 66 <= cep <= 68: return "PA"
    elif cep == 69:       return "AM_RR_AC"
    elif 70 <= cep <= 73: return "DF_GO"
    elif 74 <= cep <= 76: return "GO"
    elif cep == 77:       return "TO"
    elif cep == 78:       return "MT"
    elif cep == 79:       return "MS"
    elif 80 <= cep <= 87: return "PR"
    elif 88 <= cep <= 89: return "SC"
    elif 90 <= cep <= 99: return "RS"
    else:                 return "DESCONHECIDO"

def preparar_features(inputs, params, encoders):
    d = inputs.copy()

    # Flags de nulo
    d["FLAG_VALOR_NULO"]  = 1 if d["VALOR_A_PAGAR"] is None else 0
    d["FLAG_FUNC_NULO"]   = 1 if d["NO_FUNCIONARIOS"] is None else 0
    d["FLAG_DDD_NULO"]    = 1 if d["DDD"] is None else 0
    d["FLAG_PORTE_NULO"]  = 1 if d["PORTE"] is None else 0

    # VALOR_A_PAGAR
    valor_pagar = d["VALOR_A_PAGAR"] if d["VALOR_A_PAGAR"] else params["mediana_valor"]
    d["FLAG_VALOR_BAIXO"] = 1 if valor_pagar < params["p10_valor"] else 0
    d["VALOR_A_PAGAR"]    = min(valor_pagar, params["p99_valor"])

    # FLAG_PF
    d["FLAG_PF"] = 1 if d.get("FLAG_PF_INPUT") == "Pessoa Física" else 0

    # PORTE
    porte = d["PORTE"] or "DESCONHECIDO"
    d["FLAG_PORTE_NULO"] = 1 if d["PORTE"] is None else 0
    d["PORTE"] = porte

    # SEGMENTO
    if d.get("SEGMENTO_INDUSTRIAL") is None:
        d["SEGMENTO_INDUSTRIAL"] = (
            "PESSOA_FISICA" if d["FLAG_PF"] == 1
            else "DESCONHECIDO")

    # EMAIL
    d["DOMINIO_EMAIL"] = d.get("DOMINIO_EMAIL") or "DESCONHECIDO"

    # DDD → ESTADO
    ddd_str = str(int(d["DDD"])) if d["DDD"] else None
    d["ESTADO_DDD"] = mapa_ddd.get(ddd_str, "DESCONHECIDO")

    # CEP → REGIAO
    d["REGIAO_CEP"] = cep_para_regiao(d.get("CEP_2_DIG", 0))

    # RENDA
    renda = d.get("RENDA_MES_ANTERIOR")
    if renda is None:
        renda = params["med_renda"].get(
            porte, params["med_renda_geral"])
    d["RENDA_MES_ANTERIOR"] = min(renda, params["p99_renda"])

    # NO_FUNCIONARIOS
    func = d.get("NO_FUNCIONARIOS")
    if func is None:
        func = params["med_func"].get(
            porte, params["med_func_geral"])
    d["NO_FUNCIONARIOS"] = func

    # Features temporais
    prazo = (d["DATA_VENCIMENTO"] - d["DATA_EMISSAO"]).days
    d["PRAZO_COBRANCA"]    = prazo
    d["MES_VENCIMENTO"]    = d["DATA_VENCIMENTO"].month
    ant = (d["SAFRA_REF"] - d["DATA_CADASTRO"]).days
    ant = ant if ant >= 0 else params["med_ant"]
    d["ANTIGUIDADE_DIAS"]  = ant
    d["FLAG_CLIENTE_NOVO"] = 1 if ant < 180 else 0

    # Encoding categóricas
    categoricas = ["PORTE", "SEGMENTO_INDUSTRIAL",
                   "DOMINIO_EMAIL", "ESTADO_DDD", "REGIAO_CEP"]
    for col in categoricas:
        le = encoders[col]
        valor_col = str(d[col])
        if valor_col not in le.classes_:
            valor_col = "DESCONHECIDO"
        d[col] = int(le.transform([valor_col])[0])

    # Montar DataFrame na ordem correta
    FEATURES = [
        "VALOR_A_PAGAR", "TAXA", "PRAZO_COBRANCA",
        "MES_VENCIMENTO", "FLAG_VALOR_NULO", "FLAG_VALOR_BAIXO",
        "FLAG_PORTE_NULO", "FLAG_DDD_NULO", "FLAG_FUNC_NULO",
        "FLAG_CLIENTE_NOVO", "FLAG_PF", "PORTE",
        "SEGMENTO_INDUSTRIAL", "DOMINIO_EMAIL", "ESTADO_DDD",
        "REGIAO_CEP", "ANTIGUIDADE_DIAS", "RENDA_MES_ANTERIOR",
        "NO_FUNCIONARIOS", "N_COBR_HISTORICO",
        "TAXA_INAD_HISTORICA", "FLAG_JA_INADIMPLENTE",
        "MEDIA_DIAS_ATRASO",
    ]
    return pd.DataFrame([{f: d.get(f, 0) for f in FEATURES}])

# ═══════════════════════════════════════════════════════════════
# PÁGINA 1 — PREVISÃO INDIVIDUAL
# ═══════════════════════════════════════════════════════════════
if pagina == "💳 Previsão Individual":
    st.title("💳 Previsão de Inadimplência")
    st.markdown("Preencha os dados do cliente e da cobrança para "
                "obter a probabilidade de inadimplência.")
    st.markdown("---")

    with st.form("formulario"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📋 Dados da Cobrança")
            valor        = st.number_input(
                "Valor a Pagar (R$)", min_value=0.0,
                value=30000.0, step=1000.0)
            taxa         = st.number_input(
                "Taxa de Juros (%)", min_value=0.0,
                max_value=20.0, value=6.99, step=0.01)
            data_emissao = st.date_input(
                "Data de Emissão", value=date.today())
            data_venc    = st.date_input(
                "Data de Vencimento", value=date.today())
            safra_ref    = st.date_input(
                "Safra de Referência", value=date.today())

        with col2:
            st.subheader("🏢 Perfil do Cliente")
            porte       = st.selectbox(
                "Porte", ["Grande", "Medio", "Pequeno",
                          "Desconhecido"])
            segmento    = st.selectbox(
                "Segmento Industrial",
                ["Serviços", "Indústria", "Comércio"])
            tipo_pessoa = st.selectbox(
                "Tipo de Pessoa",
                ["Pessoa Jurídica", "Pessoa Física"])
            email       = st.selectbox(
                "Domínio do Email",
                ["Gmail", "Hotmail", "Yahoo", "Outlook",
                 "Aol", "Bol"])
            data_cadastro = st.date_input(
                "Data de Cadastro", value=date(2020, 1, 1))

        with col3:
            st.subheader("📍 Localização e Financeiro")
            ddd      = st.number_input(
                "DDD", min_value=11, max_value=99, value=11)
            cep_2dig = st.number_input(
                "CEP (2 primeiros dígitos)",
                min_value=1, max_value=99, value=13)
            renda    = st.number_input(
                "Renda/Faturamento Mês Anterior (R$)",
                min_value=0.0, value=250000.0, step=10000.0)
            n_func   = st.number_input(
                "Nº de Funcionários",
                min_value=0, value=100, step=10)

            st.subheader("📈 Histórico de Pagamentos")
            n_cobr       = st.number_input(
                "Nº de Cobranças Anteriores",
                min_value=0, value=10, step=1)
            taxa_hist    = st.slider(
                "Taxa Histórica de Inadimplência",
                0.0, 1.0, 0.05, step=0.01)
            media_atraso = st.number_input(
                "Média de Dias de Atraso Histórico",
                min_value=0.0, value=0.0, step=1.0)

        submitted = st.form_submit_button(
            "🔍 Calcular Probabilidade",
            use_container_width=True)

    if submitted:
        inputs = {
            "VALOR_A_PAGAR":        valor,
            "TAXA":                 taxa,
            "DATA_EMISSAO":         pd.Timestamp(data_emissao),
            "DATA_VENCIMENTO":      pd.Timestamp(data_venc),
            "SAFRA_REF":            pd.Timestamp(safra_ref),
            "DATA_CADASTRO":        pd.Timestamp(data_cadastro),
            "PORTE":                porte,
            "SEGMENTO_INDUSTRIAL":  segmento,
            "FLAG_PF_INPUT":        tipo_pessoa,
            "DOMINIO_EMAIL":        email,
            "DDD":                  ddd,
            "CEP_2_DIG":            cep_2dig,
            "RENDA_MES_ANTERIOR":   renda,
            "NO_FUNCIONARIOS":      n_func,
            "N_COBR_HISTORICO":     n_cobr,
            "TAXA_INAD_HISTORICA":  taxa_hist,
            "FLAG_JA_INADIMPLENTE": 1 if taxa_hist > 0 else 0,
            "MEDIA_DIAS_ATRASO":    media_atraso,
        }

        X = preparar_features(inputs, params, encoders)
        prob = float(modelo.predict_proba(X)[0][1])
        faixa, cor, acao, detalhe, prioridade = classificar_risco(prob)

        st.markdown("---")
        st.subheader("📊 Resultado da Previsão")

        col_prob, col_faixa, col_acao = st.columns(3)

        with col_prob:
            st.metric(
                label="Probabilidade de Inadimplência",
                value=f"{prob:.1%}")
            st.progress(prob)

        with col_faixa:
            st.markdown("**Faixa de Risco**")
            st.markdown(
                f"<h2 style='color:{cor}'>{faixa}</h2>",
                unsafe_allow_html=True)
            st.markdown(f"**Prioridade:** {prioridade}")

        with col_acao:
            st.markdown("**Ação Recomendada**")
            mensagem = f"**{acao}**\n\n{detalhe}"
            st.info(mensagem)

        # Indicador de confiança
        st.markdown("---")
        if n_cobr == 0:
            st.warning(
                "⚠️ **Baixa confiança** — cliente sem histórico "
                "de pagamentos. O modelo depende apenas do perfil "
                "cadastral e da cobrança.")
        elif n_cobr <= 3:
            st.warning(
                "⚠️ **Confiança moderada** — cliente com histórico "
                f"limitado ({n_cobr} cobranças anteriores).")
        else:
            st.success(
                f"✅ **Alta confiança** — cliente com {n_cobr} "
                "cobranças anteriores no histórico.")

        # Top 3 fatores SHAP
        st.markdown("---")
        st.subheader("🔍 Principais Fatores da Previsão")

        try:
            modelo_base = modelo.estimator
            explainer   = shap.TreeExplainer(modelo_base)
            shap_vals   = explainer.shap_values(X)

            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            else:
                sv = shap_vals[0]

            FEATURES_LIST = list(X.columns)
            shap_df = pd.DataFrame({
                "feature": FEATURES_LIST,
                "shap":    sv
            }).iloc[pd.Series(sv).abs().sort_values(
                ascending=False).index].head(3)

            for _, row in shap_df.iterrows():
                direcao = "aumenta" if row["shap"] > 0 else "diminui"
                cor_dir = "#DC2626" if row["shap"] > 0 else "#16A34A"
                st.markdown(
                    f"<span style='color:{cor_dir}'>●</span> "
                    f"**{row['feature']}** {direcao} o risco "
                    f"(impacto: {abs(row['shap']):.3f})",
                    unsafe_allow_html=True)
        except Exception:
            st.info("Análise SHAP não disponível para este caso.")

# ═══════════════════════════════════════════════════════════════
# PÁGINA 2 — PERFORMANCE DO MODELO
# ═══════════════════════════════════════════════════════════════
elif pagina == "📊 Performance do Modelo":
    st.title("📊 Performance do Modelo")
    st.markdown("LightGBM com calibração isotônica — "
                "validação em jan–jun/2021.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC",     "0.9444", "↑ excelente")
    c2.metric("Gini",        "0.8888", "↑ excelente")
    c3.metric("KS",          "0.7573", "↑ excelente")
    c4.metric("Brier Score", "0.0362", "↓ bem calibrado")

    st.markdown("---")

    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.subheader("Curva ROC")
        y_val_arr  = np.array(validacao["y_val"])
        y_prob_arr = np.array(validacao["prob_final"])
        auc        = roc_auc_score(y_val_arr, y_prob_arr)
        fpr, tpr, _ = roc_curve(y_val_arr, y_prob_arr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#1E2761", linewidth=2,
                label=f"LightGBM (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="#94A3B8",
                linestyle="--", linewidth=1,
                label="Aleatório (AUC = 0.5)")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#1E2761")
        ax.set_xlabel("Taxa de Falsos Positivos")
        ax.set_ylabel("Taxa de Verdadeiros Positivos")
        ax.set_title("Curva ROC — LightGBM Calibrado")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col_cm:
        st.subheader("Threshold Dinâmico")
        threshold = st.slider(
            "Ajuste o threshold",
            min_value=0.05, max_value=0.60,
            value=0.2389, step=0.01,
            help="Mova para ver o impacto nas métricas")

        y_pred = (y_prob_arr >= threshold).astype(int)
        cm     = confusion_matrix(y_val_arr, y_pred)
        prec   = precision_score(y_val_arr, y_pred, zero_division=0)
        rec    = recall_score(y_val_arr, y_pred, zero_division=0)
        f1     = f1_score(y_val_arr, y_pred, zero_division=0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precisão", f"{prec:.1%}")
        m2.metric("Recall",   f"{rec:.1%}")
        m3.metric("F1-Score", f"{f1:.3f}")

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        cores_cm = [["#16A34A", "#EA580C"],
                    ["#DC2626", "#16A34A"]]
        labels_cm = [["VN", "FP"], ["FN", "VP"]]
        for i in range(2):
            for j in range(2):
                ax2.add_patch(mpatches.Rectangle(
                    (j, 1-i), 1, 1,
                    color=cores_cm[i][j], alpha=0.8))
                ax2.text(j+0.5, 1.5-i, f"{cm[i][j]:,}",
                         ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white")
                ax2.text(j+0.5, 1.2-i, labels_cm[i][j],
                         ha="center", va="center",
                         fontsize=10, color="white")

        ax2.set_xlim(0, 2)
        ax2.set_ylim(0, 2)
        ax2.set_xticks([0.5, 1.5])
        ax2.set_yticks([0.5, 1.5])
        ax2.set_xticklabels(["Adimplente", "Inadimplente"])
        ax2.set_yticklabels(["Inadimplente", "Adimplente"])
        ax2.set_xlabel("Predito")
        ax2.set_ylabel("Real")
        ax2.set_title(
            f"Matriz de Confusão (threshold={threshold:.2f})")
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.info(
        f"**Threshold atual: {threshold:.4f}** — "
        f"De {cm[1][0]+cm[1][1]:,} inadimplentes reais, "
        f"**{cm[1][1]:,} detectados ({rec:.1%} de recall)** "
        f"e {cm[1][0]:,} não detectados. "
        f"De {cm[0][0]+cm[0][1]:,} adimplentes, "
        f"**{cm[0][1]:,} falsos alarmes "
        f"({cm[0][1]/(cm[0][0]+cm[0][1]):.1%})**."
    )

# ═══════════════════════════════════════════════════════════════
# PÁGINA 3 — SOBRE O PROJETO
# ═══════════════════════════════════════════════════════════════
elif pagina == "ℹ️ Sobre o Projeto":
    st.title("ℹ️ Sobre o Projeto")
    st.markdown("---")

    st.subheader("Contexto")
    st.markdown(
         "Este projeto simula um sistema de inteligência de crédito "
         "para uma instituição financeira fictícia. O desafio consiste "
         "em identificar, com antecedência, quais clientes têm maior "
         "probabilidade de não pagar uma cobrança mensal no prazo — "
         "permitindo que a equipe de cobrança aja de forma proativa "
         "antes que a inadimplência se concretize. "
         "A solução foi construída sobre um histórico real de transações "
         "financeiras, combinando análise exploratória aprofundada, "
         "engenharia de features comportamentais e um modelo preditivo "
         "calibrado para gerar probabilidades confiáveis entre 0 e 1.")

    st.subheader("Definição de Inadimplência")
    st.info("Pagamento realizado com **5 ou mais dias de atraso** "
            "em relação à data de vencimento → TARGET = 1")

    st.subheader("Pipeline do Projeto")
    etapas = [
        ("1. Inspeção dos Dados",
         "Análise de qualidade, nulos e consistência das 4 bases"),
        ("2. Construção do Target",
         "Regra dos 5 dias aplicada sobre DATA_PAGAMENTO"),
        ("3. EDA Aprofundada",
         "6 insights principais que fundamentaram as decisões"),
        ("4. Feature Engineering",
         "23 features criadas sem data leakage"),
        ("5. Modelagem",
         "5 modelos comparados via time-based split"),
        ("6. Otimização",
         "Optuna com 100 trials — AUC: 0.9291 → 0.9417"),
        ("7. Calibração",
         "Isotonic Regression — Brier: 0.0503 → 0.0362 (−28%)"),
        ("8. Submissão",
         "11.542 previsões geradas para a base de teste"),
    ]
    for titulo, desc in etapas:
        st.markdown(f"**{titulo}** — {desc}")

    st.subheader("Features do Modelo")
    grupos = {
        "Histórico Individual": [
            "TAXA_INAD_HISTORICA", "N_COBR_HISTORICO",
            "FLAG_JA_INADIMPLENTE", "MEDIA_DIAS_ATRASO"
        ],
        "Cobrança Atual": [
            "VALOR_A_PAGAR", "TAXA", "PRAZO_COBRANCA",
            "MES_VENCIMENTO", "FLAG_VALOR_BAIXO", "FLAG_VALOR_NULO"
        ],
        "Perfil Cadastral": [
            "PORTE", "FLAG_PORTE_NULO", "SEGMENTO_INDUSTRIAL",
            "ESTADO_DDD", "FLAG_DDD_NULO", "REGIAO_CEP",
            "FLAG_PF", "DOMINIO_EMAIL", "ANTIGUIDADE_DIAS",
            "FLAG_CLIENTE_NOVO"
        ],
        "Comportamento Mensal": [
            "RENDA_MES_ANTERIOR", "NO_FUNCIONARIOS", "FLAG_FUNC_NULO"
        ],
    }

    cols = st.columns(2)
    for i, (grupo, features) in enumerate(grupos.items()):
        with cols[i % 2]:
            st.markdown(f"**{grupo}**")
            for f in features:
                st.markdown(f"- {f}")

    st.subheader("Referências")
    st.markdown("""
- Ke et al. (2017). LightGBM. NeurIPS.
- Lundberg & Lee (2017). SHAP. NeurIPS.
- Akiba et al. (2019). Optuna. KDD.
- Siddiqi (2012). Credit Risk Scorecards. Wiley.
""")

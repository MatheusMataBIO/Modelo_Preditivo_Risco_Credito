# Modelo Preditivo de Risco de Crédito

Sistema de inteligência de crédito para previsão de inadimplência
em cobranças mensais. O modelo estima a probabilidade de um cliente
não realizar o pagamento no prazo, permitindo ações proativas de cobrança.

---

## Demonstração

🚀 **App em produção:** [Acesse aqui](https://modelopreditivoriscocredito-vomd3tjf8hbm5ayyeuzxho.streamlit.app/)

---

## Problema

Uma instituição financeira realiza cobranças mensais e precisa identificar,
com antecedência, quais clientes têm maior risco de atrasar o pagamento
em 5 ou mais dias. Essa informação é usada para priorizar ações proativas
de cobrança e reduzir a inadimplência da carteira.

**Definição de inadimplência:** pagamento realizado com 5 ou mais dias
de atraso em relação à data de vencimento → TARGET = 1

---

## Resultados do Modelo

| Métrica    | Valor  | Referência de mercado |
|------------|--------|----------------------|
| AUC-ROC    | 0.9444 | > 0.75 bom           |
| Gini       | 0.8888 | > 0.60 bom           |
| KS         | 0.7573 | > 0.40 bom           |
| Brier Score| 0.0362 | quanto menor melhor  |
| Recall     | 65.6%  | threshold = 0.2389   |
| Precisão   | 55.4%  | threshold = 0.2389   |

---

## Pipeline do Projeto

**Inspeção e Qualidade dos Dados**
└── Análise de nulos, duplicatas e consistência das bases


**Construção do Target**
└── Regra dos 5 dias sobre DATA_PAGAMENTO - DATA_VENCIMENTO


**Análise Exploratória (EDA)**
└── 6 insights principais que fundamentaram as decisões


**Feature Engineering**
└── 23 features sem data leakage


**Modelagem**
└── 5 modelos comparados via time-based split


**Otimização**
└── Optuna — 100 trials — AUC: 0.9291 → 0.9417


**Calibração**
└── Isotonic Regression — Brier: 0.0503 → 0.0362 (−28%)


**Deploy**
└── Streamlit Cloud com previsão em tempo real


---

## Principais Insights da EDA

| # | Insight | Impacto |
|---|---------|---------|
| 1 | Clientes que já atrasaram têm taxa de 13.7% vs 1.4% | 10x mais risco |
| 2 | 13% dos clientes respondem por 55% das inadimplências | Concentração de risco |
| 3 | Cobranças abaixo de R$12.552 têm taxa de 30.3% | Sinal mais forte da EDA |
| 4 | Norte/Centro-Oeste acima de 16%, Sul abaixo de 5% | Padrão geográfico claro |
| 5 | Clientes sem porte têm taxa de 17.5% vs 6.7% | Proxy de menor formalização |
| 6 | Dezembro (7.9%) e Maio (7.6%) acima da média | Sazonalidade real |

---

## Features do Modelo

### Histórico Individual — as mais importantes
| Feature | Descrição |
|---------|-----------|
| TAXA_INAD_HISTORICA | % de cobranças anteriores em atraso |
| N_COBR_HISTORICO | Nº de cobranças anteriores |
| FLAG_JA_INADIMPLENTE | Já atrasou alguma vez |
| MEDIA_DIAS_ATRASO | Média histórica de dias de atraso |

### Cobrança Atual
| Feature | Descrição |
|---------|-----------|
| VALOR_A_PAGAR | Valor com clip no p99 |
| FLAG_VALOR_BAIXO | Cobrança abaixo do p10 — risco 6.7x maior |
| FLAG_VALOR_NULO | Nulo informativo — taxa 9.4% vs 7.0% |
| TAXA | Taxa de juros da cobrança |
| PRAZO_COBRANCA | Dias entre emissão e vencimento |
| MES_VENCIMENTO | Sazonalidade — dez e mai acima da média |

### Perfil Cadastral
| Feature | Descrição |
|---------|-----------|
| PORTE + FLAG_PORTE_NULO | Sem porte tem taxa 17.5% |
| SEGMENTO_INDUSTRIAL | Serviços vs Indústria vs Comércio |
| ESTADO_DDD + FLAG_DDD_NULO | Geografia por estado |
| REGIAO_CEP | Geografia por região do CEP |
| FLAG_PF | Pessoa física tem taxa 20.1% vs 7% PJ |
| DOMINIO_EMAIL | Hotmail 9.3% vs AOL 3.9% |
| ANTIGUIDADE_DIAS + FLAG_CLIENTE_NOVO | Clientes novos com taxa 12.5% |

### Comportamento Mensal
| Feature | Descrição |
|---------|-----------|
| RENDA_MES_ANTERIOR | Faturamento do mês anterior — clip p99 |
| NO_FUNCIONARIOS + FLAG_FUNC_NULO | Nulo informativo — 8.3% vs 6.9% |

---

## Decisões Técnicas

**Por que time-based split?**
Dados financeiros têm estrutura temporal. KFold tradicional causaria
data leakage — o modelo treinaria em cobranças futuras para prever
cobranças passadas. Treino até dez/2020, validação em jan–jun/2021.

**Por que LightGBM?**
O Random Forest teve AUC maior (0.9409), mas Brier Score 3x pior
(0.1072 vs 0.0362). Como o objetivo é gerar probabilidades confiáveis
entre 0 e 1, a qualidade da probabilidade foi o critério decisivo.

**Por que calibração isotônica?**
Sem calibração, um score de 0.20 correspondia a risco real de ~40%.
A Isotonic Regression reduziu o Brier Score em 28% sem prejudicar o AUC.

**Como foi evitado o data leakage?**
Features de histórico calculadas com `shift(1)` + `expanding()` —
cada cobrança usa apenas cobranças anteriores. Parâmetros de imputação
calculados exclusivamente no treino e aplicados no teste.

---

## Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44.1-purple)
![Optuna](https://img.shields.io/badge/Optuna-3.6.1-blue)

| Biblioteca | Uso |
|------------|-----|
| LightGBM | Modelo principal |
| Scikit-learn | Pipeline, métricas e calibração |
| Optuna | Otimização bayesiana de hiperparâmetros |
| SHAP | Interpretabilidade do modelo |
| Pandas / NumPy | Manipulação de dados |
| Matplotlib | Visualizações |
| Streamlit | Interface web |
| Joblib | Serialização do modelo |

---

## Como Executar Localmente

```bash
# Clonar o repositório
git clone https://github.com/MatheusMataBIO/Modelo_Preditivo_Risco_Credito.git
cd Modelo_Preditivo_Risco_Credito

# Instalar dependências
pip install -r requirements.txt

# Rodar o app
streamlit run app.py
```

---

## Referências

- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient
  Boosting Decision Tree*. NeurIPS.
- Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to
  Interpreting Model Predictions*. NeurIPS.
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter
  Optimization Framework*. KDD.
- Siddiqi, N. (2012). *Credit Risk Scorecards*. Wiley.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good
  Probabilities with Supervised Learning*. ICML.


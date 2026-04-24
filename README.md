# TCC

# Dashboard de Predição de Risco em Hemodiálise

> Documentação técnica — `src/dashboard.py`  
> Universidade Federal de São Carlos (UFSCar) · Trabalho de Conclusão de Curso  
> Autor: Cauã Benini da Silva

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Estrutura de Diretórios](#2-estrutura-de-diretórios)
3. [Instalação e Execução](#3-instalação-e-execução)
4. [Interface do Dashboard](#4-interface-do-dashboard)
   - 4.1 [Barra Lateral (Sidebar)](#41-barra-lateral-sidebar)
   - 4.2 [Entrada de Dados Clínicos](#42-entrada-de-dados-clínicos)
   - 4.3 [Resultados e Visualizações](#43-resultados-e-visualizações)
5. [Simulação de Valores Faltantes via KNN de Trajetória](#5-simulação-de-valores-faltantes-via-knn-de-trajetória)
   - 5.1 [Motivação](#51-motivação)
   - 5.2 [Algoritmo passo a passo](#52-algoritmo-passo-a-passo)
   - 5.3 [Parâmetros configuráveis](#53-parâmetros-configuráveis)
   - 5.4 [Exemplo ilustrativo](#54-exemplo-ilustrativo)
6. [Cálculo da Confiança da Predição](#6-cálculo-da-confiança-da-predição)
   - 6.1 [Saída do modelo](#61-saída-do-modelo)
   - 6.2 [Tabela de métricas](#62-tabela-de-métricas)
   - 6.3 [Interpretação das faixas de risco](#63-interpretação-das-faixas-de-risco)
7. [Modelos Disponíveis](#7-modelos-disponíveis)
8. [Exportação de Relatório](#8-exportação-de-relatório)
9. [Variáveis Clínicas](#9-variáveis-clínicas)
10. [Decisões Técnicas e Limitações](#10-decisões-técnicas-e-limitações)

---

## 1. Visão Geral

O dashboard tem como objetivo apoiar profissionais de saúde na **predição de hipotensão intradiálica** (TARGET = 1) durante sessões de hemodiálise. A ferramenta combina duas etapas principais:

1. **Imputação de valores faltantes** — quando o usuário fornece apenas medições parciais da sessão (por exemplo, apenas a primeira hora), os valores das horas restantes são estimados automaticamente por meio de um algoritmo de KNN de Trajetória com Ponderação por Distância.

2. **Predição de risco** — com o vetor clínico completo (H0 a H5), um modelo de *machine learning* pré-treinado classifica a sessão como de alto risco (TARGET = 1) ou baixo risco (TARGET = 0) e, quando possível, fornece a probabilidade associada.

> **Aviso clínico:** todas as predições são de caráter exclusivamente de suporte à decisão e devem ser revisadas por profissional de saúde habilitado.

---

## 2. Estrutura de Diretórios

```
TCC/
├── data/
│   ├── dataset.csv            # Dataset original (bruto)
│   └── dataset_flat.csv       # Dataset pré-processado (formato sessão × atributos)
├── models/
│   ├── modelo_knn.pkl
│   ├── modelo_RF.pkl
│   ├── modelo_svm.pkl
│   ├── modelo_xgboost.pkl
│   ├── modelo_DT.pkl
│   ├── modelo_MLP.pkl
│   └── modelo_NB.pkl
├── src/
│   └── dashboard.py           # Aplicação Streamlit
├── requirements.txt
└── README.md
```

---

## 3. Instalação e Execução

### Pré-requisitos

- Python 3.9 ou superior
- Dependências listadas em `requirements.txt`

### Instalação

```bash
# Na raiz do projeto
pip install -r requirements.txt
```

### Execução

```bash
streamlit run src/dashboard.py
```

O dashboard será aberto automaticamente no navegador em `http://localhost:8501`.

### Re-serialização dos modelos (caso ocorra erro de versão)

Se aparecer o erro `STACK_GLOBAL requires str` ao carregar um modelo, re-salve todos os `.pkl` no notebook de treinamento usando `joblib`, que é o serializador recomendado pelo scikit-learn:

```python
import joblib

joblib.dump(modelo_knn,     'models/modelo_knn.pkl')
joblib.dump(modelo_RF,      'models/modelo_RF.pkl')
joblib.dump(modelo_svm,     'models/modelo_svm.pkl')
joblib.dump(modelo_xgboost, 'models/modelo_xgboost.pkl')
joblib.dump(modelo_DT,      'models/modelo_DT.pkl')
joblib.dump(modelo_MLP,     'models/modelo_MLP.pkl')
joblib.dump(modelo_NB,      'models/modelo_NB.pkl')
```

---

## 4. Interface do Dashboard

### 4.1 Barra Lateral (Sidebar)

| Controle | Descrição |
|---|---|
| **Sexo** | Sexo do paciente: `Female (0)` ou `Male (1)` |
| **Idade** | Idade em anos (inteiro, 1–120) |
| **Modelo** | Lista suspensa com os 7 modelos pré-treinados disponíveis |
| **K-Vizinhos** | Número de vizinhos utilizados na imputação KNN (padrão: 10, intervalo: 3–50) |
| **Incluir demografia** | Toggle para incluir SEX e AGE no cálculo de distância do KNN |
| **Upload CSV** | Carregamento opcional de um arquivo `.csv` com dados pré-preenchidos no mesmo formato do `dataset_flat.csv` |

### 4.2 Entrada de Dados Clínicos

A seção principal exibe uma grade de entrada com:

- **Checkboxes de hora (H0–H5):** ativam ou desativam a entrada de dados para aquela hora. H0 vem marcada por padrão. Horas desmarcadas serão **simuladas** automaticamente.
- **Expanders por variável:** cada uma das 12 variáveis clínicas possui um expander com campos numéricos para cada hora habilitada. Os campos respeitam os limites fisiológicos de cada variável.

Ao clicar em **▶ Simulate & Predict**, o dashboard executa a simulação e a predição.

### 4.3 Resultados e Visualizações

Após a execução, são exibidos:

- **Alerta de risco** com cor e ícone indicativos (🔴 alto risco / 🟢 baixo risco)
- **Métricas resumidas** (TARGET, probabilidade, confiança, modelo)
- **Tabela H0–H5** com marcação visual de valores observados (●, azul) e simulados (◌, verde)
- **Gráficos interativos** de trajetória por categoria clínica (Hemodinâmica, Pressões, Fluxo, Volume, Banho)
- **Gauge de probabilidade** com faixas de risco e tabela de métricas detalhada
- **Botão de exportação** de relatório CSV

---

## 5. Simulação de Valores Faltantes via KNN de Trajetória

### 5.1 Motivação

Em contexto clínico real, é comum que apenas as medições das primeiras horas de uma sessão estejam disponíveis no momento da tomada de decisão. Para que o modelo preditivo possa ser aplicado mesmo assim — pois ele requer o vetor completo de H0 a H5 —, o dashboard implementa uma estratégia de imputação baseada em **similaridade de trajetória entre sessões históricas**.

A abordagem é distinta de uma imputação simples por média global: em vez de preencher valores faltantes com estatísticas agregadas do dataset inteiro, busca-se as sessões passadas mais parecidas com a sessão atual (com base nos valores já observados) e estima-se as horas faltantes como uma média ponderada dessas sessões análogas.

### 5.2 Algoritmo passo a passo

**Entrada:**
- Valores observados para um subconjunto de horas (ex.: apenas H0, ou H0+H1)
- Sexo e idade do paciente
- Dataset histórico `dataset_flat.csv` com 13.895 sessões completas

**Passo 1 — Construção do vetor de busca**

Apenas as colunas correspondentes às horas e variáveis **efetivamente observadas** são usadas para comparação. Se `use_demographics = True`, SEX e AGE também são incluídos.

```
vetor_busca = [SEX, AGE, SBP_H0, DBP_H0, HRA_H0, ...]
               ↑ apenas colunas com valor fornecido pelo usuário
```

**Passo 2 — Normalização**

O vetor de busca e todas as linhas do dataset histórico são normalizados pelo mesmo `StandardScaler` (média zero, desvio padrão um). Isso garante que variáveis com escalas muito diferentes (ex.: BFR em mL/min vs. IWG em kg) contribuam igualmente para o cálculo de distância.

**Passo 3 — Busca dos K vizinhos mais próximos**

Aplica-se o algoritmo KNN com métrica Euclidiana no espaço normalizado das horas observadas. O resultado são os K índices e distâncias das sessões históricas mais similares.

```
distâncias, índices = NearestNeighbors(k).kneighbors(vetor_busca_normalizado)
```

**Passo 4 — Ponderação inversa pela distância**

Cada vizinho recebe um peso inversamente proporcional à sua distância à sessão atual. Vizinhos mais próximos (mais similares) têm maior influência na estimativa:

```
peso_i = 1 / (distância_i + ε)      (ε = 1×10⁻⁸ para evitar divisão por zero)
peso_i = peso_i / Σ pesos            (normalização para soma = 1)
```

**Passo 5 — Estimativa das horas faltantes**

Para cada variável e cada hora que não foi fornecida pelo usuário, calcula-se a média ponderada dos K vizinhos:

```
valor_estimado(VAR, Hx) = Σ ( peso_i × valor_vizinho_i(VAR, Hx) )
```

As horas já observadas **nunca são alteradas** — apenas as faltantes recebem estimativas.

**Saída:**
Um dicionário completo `{variável: {hora: valor}}` para todas as 12 variáveis × 6 horas, pronto para ser convertido no vetor de entrada do modelo preditivo.

### 5.3 Parâmetros configuráveis

| Parâmetro | Controle na UI | Padrão | Efeito |
|---|---|---|---|
| `k` | Slider *K-Vizinhos* | 10 | Mais K → estimativa mais suavizada; menos K → mais sensível a sessões individuais |
| `use_demographics` | Toggle *Incluir demografia* | `True` | Inclui SEX e AGE no espaço de busca; útil quando perfil demográfico é relevante para o padrão clínico |

### 5.4 Exemplo ilustrativo

Suponha que o usuário forneça apenas SBP_H0 = 145 e DBP_H0 = 88. O algoritmo:

1. Busca no histórico as 10 sessões em que SBP_H0 ≈ 145 e DBP_H0 ≈ 88 (considerando também SEX e AGE, se habilitado).
2. Calcula os pesos inversamente proporcionais às distâncias dessas 10 sessões.
3. Para SBP_H1, por exemplo, calcula: `SBP_H1 = 0.38×142 + 0.21×139 + 0.14×144 + ... = 141.3`
4. Repete o processo para todas as 12 variáveis × 5 horas restantes (H1–H5).

O resultado é uma trajetória completa coerente com o padrão histórico de pacientes similares, e não uma simples replicação do valor de H0.

---

## 6. Cálculo da Confiança da Predição

### 6.1 Saída do modelo

Após a simulação, o vetor completo (SEX, AGE + 72 atributos clínicos H0–H5) é passado ao modelo pré-treinado selecionado. O modelo retorna:

- **`predict(X)`** — classe predita: `0` (sem hipotensão) ou `1` (hipotensão intradiálica)
- **`predict_proba(X)`** — vetor de probabilidades `[P(TARGET=0), P(TARGET=1)]`, disponível em todos os modelos exceto SVM linear puro

As probabilidades são estimadas diretamente pelo modelo a partir da estrutura aprendida durante o treinamento. No caso do KNN, por exemplo, `P(TARGET=1)` equivale à fração de vizinhos de treino que pertencem à classe 1; no caso de modelos probabilísticos como Naive Bayes e MLP, deriva-se da função softmax ou da distribuição posterior.

### 6.2 Tabela de métricas

A seção de predição exibe a seguinte tabela de métricas para cada execução:

| Métrica | Valor |
|---|---|
| P(TARGET=1) | probabilidade de hipotensão intradiálica |
| P(TARGET=0) | probabilidade de sessão sem hipotensão |
| Predicted class | classe predita (0 ou 1) |
| Model | nome do modelo selecionado |

**Exemplo de saída:**

| Métrica | Valor |
|---|---|
| P(TARGET=1) | `0.6135` |
| P(TARGET=0) | `0.3865` |
| Predicted class | `1` |
| Model | `XGBoost` |

Neste exemplo, o modelo XGBoost atribui 61,35% de probabilidade de ocorrência de hipotensão intradiálica, classificando a sessão como de **alto risco** (TARGET = 1).

A **confiança** exibida no card de métricas é calculada como:

```
Confiança = max( P(TARGET=0), P(TARGET=1) )
```

Ou seja, reflete o quão certo o modelo está sobre sua própria decisão, independentemente da classe predita. No exemplo acima: `max(0.3865, 0.6135) = 61,35%`.

### 6.3 Interpretação das faixas de risco

O gauge exibido no dashboard usa as seguintes faixas de P(TARGET=1):

| Faixa | Classificação | Recomendação |
|---|---|---|
| 0% – 30% | 🟢 **BAIXO** | Sessão de rotina esperada |
| 30% – 60% | 🟡 **MODERADO** | Monitoramento padrão |
| 60% – 80% | 🟠 **ALTO** | Vigilância aumentada recomendada |
| 80% – 100% | 🔴 **CRÍTICO** | Monitoramento imediato necessário |

> **Nota:** modelos sem suporte a `predict_proba` (como alguns SVMs com kernel não-linear) retornam apenas a classe predita, sem probabilidade. Nesses casos, o gauge não é exibido.

---

## 7. Modelos Disponíveis

| Nome no Dashboard | Arquivo | Algoritmo |
|---|---|---|
| K-Nearest Neighbor (KNN) | `modelo_knn.pkl` | KNN classificador |
| Random Forest (RF) | `modelo_RF.pkl` | Ensemble de árvores de decisão |
| Support Vector Machine (SVM) | `modelo_svm.pkl` | SVM com kernel RBF |
| XGBoost | `modelo_xgboost.pkl` | Gradient Boosting extremo |
| Decision Tree (DT) | `modelo_DT.pkl` | Árvore de decisão simples |
| Multi-Layer Perceptron (MLP) | `modelo_MLP.pkl` | Rede neural densa |
| Naive Bayes (NB) | `modelo_NB.pkl` | Classificador probabilístico bayesiano |

Todos os modelos foram treinados sobre o `dataset_flat.csv` com amostragem holdout estratificada (80% treino / 20% teste), com estratificação pelo TARGET para manutenção da proporção de classes.

---

## 8. Exportação de Relatório

Ao final de cada execução, o botão **⬇ Download CSV Report** gera um arquivo `.csv` com as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `Variable` | Nome da variável clínica (ex.: `SBP`) |
| `Hour` | Hora da sessão (H0–H5) |
| `Value` | Valor observado ou simulado |
| `Source` | `Observed` (inserido pelo usuário) ou `Simulated` (estimado pelo KNN) |
| `SEX` | Sexo codificado (0 = feminino, 1 = masculino) |
| `AGE` | Idade em anos |
| `Prediction_TARGET` | Classe predita (0 ou 1) |
| `Probability_TARGET1` | P(TARGET=1), ou vazio se o modelo não suportar |
| `Model` | Nome do modelo utilizado |
| `K_neighbors` | Valor de K usado na imputação |

O nome do arquivo segue o padrão `hd_session_SEX{n}_AGE{n}.csv`.

---

## 9. Variáveis Clínicas

| Sigla | Parâmetro | Unidade |
|---|---|---|
| IWG | Ganho de peso interdialítico | kg |
| VOL | Alterações de volume | L |
| KT | Depuração de ureia | L |
| BFR | Fluxo sanguíneo | mL/min |
| HBC | Condutividade do banho de hemodiálise | mScm |
| APR | Pressão arterial pré-bomba | mmHg |
| VPR | Pressão venosa pós-filtro | mmHg |
| TMP | Pressão transmembrana | mmHg |
| SBP | Pressão arterial sistólica | mmHg |
| DBP | Pressão arterial diastólica | mmHg |
| HRA | Frequência cardíaca | bpm |
| TUF | Ultrafiltração total | mL |

> **Nota sobre APR:** valores negativos são fisiologicamente esperados na hemodiálise, pois representam a pressão de sucção gerada pela bomba na via arterial do circuito extracorpóreo.

---

## 10. Decisões Técnicas e Limitações

### Identificação de pacientes

O dataset não possui identificadores individuais explícitos. Seguindo o critério do artigo base (*Predicting the Appearance of Hypotension during Hemodialysis Sessions Using Machine Learning Classifiers*, MDPI, 2021), a combinação **SEX + AGE** é utilizada como proxy de identificação de paciente.

### Definição do TARGET

Um episódio de hipotensão intradiálica é definido como a queda da pressão arterial sistólica ≥ 20 mmHg entre os períodos iniciais e intermediários da sessão. A variável TARGET = 1 indica que ao menos uma medição da sessão satisfez esse critério.

### Limitações conhecidas

- **Identificação por SEX+AGE:** pacientes com mesmo sexo e idade são tratados como o mesmo indivíduo, o que pode introduzir ruído nas sessões históricas utilizadas na busca KNN.
- **Simulação vs. medição real:** os valores imputados são estimativas estatísticas e não substituem medições clínicas reais. A qualidade da simulação depende da representatividade do histórico disponível.
- **Generalização do modelo:** os modelos foram treinados em dados de um único hospital (Hospital Príncipe de Asturias, Madrid, 2016–2019) e podem não generalizar diretamente para outras populações ou protocolos de diálise.
- **Ausência de validação prospectiva:** os resultados devem ser interpretados como suporte à decisão clínica, não como diagnóstico definitivo.

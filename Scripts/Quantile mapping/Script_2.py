import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================================
# CAMINHOS
# ==========================================================

# Caminho para o ficheiro parquet de treino de radiação gerado no passo anterior
TREINO = r""

PASTA_SAIDA = r""

os.makedirs(PASTA_SAIDA, exist_ok=True)

# ==========================================================
# PARÂMETROS
# ==========================================================

MIN_OBS = 30

print("=" * 60)
print("TREINO EMPIRICAL QUANTILE MAPPING - RADIAÇÃO SOLAR")
print("=" * 60)

# ==========================================================
# LER DADOS
# ==========================================================

if not os.path.exists(TREINO):
    raise FileNotFoundError(f"O ficheiro de treino não foi encontrado em: {TREINO}")

df = pd.read_parquet(TREINO)

print(f"Observações : {len(df):,}")
print(f"Estações    : {df.Estacao.nunique()}")

# ==========================================================
# MODELOS
# ==========================================================

modelos = {}

estacoes = sorted(df.Estacao.unique())

# ==========================================================
# LOOP DAS ESTAÇÕES
# ==========================================================

for estacao in tqdm(estacoes):

    dados = df[df["Estacao"] == estacao].copy()

    dados = dados.dropna(subset=["ERA5", "OBS"])

    if len(dados) < MIN_OBS:
        continue

    era = dados["ERA5"].astype(float).values
    obs = dados["OBS"].astype(float).values

    # Garantir limites físicos da radiação (radiação não pode ser negativa)
    era = np.maximum(era, 0.0)
    obs = np.maximum(obs, 0.0)

    # ======================================================
    # DISTRIBUIÇÕES EMPÍRICAS
    # ======================================================

    era = np.sort(era)
    obs = np.sort(obs)

    prob_era = (
        np.arange(len(era)) + 0.5
    ) / len(era)

    prob_obs = (
        np.arange(len(obs)) + 0.5
    ) / len(obs)

    # ======================================================
    # GUARDAR MODELO DA ESTAÇÃO
    # ======================================================

    modelos[estacao] = {

        # ---------------------------------------------
        # Identificação
        # ---------------------------------------------

        "Estacao": estacao,

        "Codigo": dados["Codigo"].iloc[0],

        "Latitude": float(dados["Latitude"].iloc[0]),

        "Longitude": float(dados["Longitude"].iloc[0]),

        # ---------------------------------------------
        # Informação geral
        # ---------------------------------------------

        "N": int(len(dados)),

        # ---------------------------------------------
        # Distribuições empíricas
        # ---------------------------------------------

        "ERA5": era.astype(np.float32),

        "OBS": obs.astype(np.float32),

        "CDF_ERA5": prob_era.astype(np.float32),

        "CDF_OBS": prob_obs.astype(np.float32),

        # ---------------------------------------------
        # Estatísticas
        # ---------------------------------------------

        "Media_ERA5": float(np.mean(era)),

        "Media_OBS": float(np.mean(obs)),

        "Mediana_ERA5": float(np.median(era)),

        "Mediana_OBS": float(np.median(obs)),

        "Std_ERA5": float(np.std(era)),

        "Std_OBS": float(np.std(obs)),

        "Min_ERA5": float(np.min(era)),

        "Min_OBS": float(np.min(obs)),

        "Max_ERA5": float(np.max(era)),

        "Max_OBS": float(np.max(obs)),

        "P5_ERA5": float(np.percentile(era, 5)),

        "P5_OBS": float(np.percentile(obs, 5)),

        "P25_ERA5": float(np.percentile(era, 25)),

        "P25_OBS": float(np.percentile(obs, 25)),

        "P50_ERA5": float(np.percentile(era, 50)),

        "P50_OBS": float(np.percentile(obs, 50)),

        "P75_ERA5": float(np.percentile(era, 75)),

        "P75_OBS": float(np.percentile(obs, 75)),

        "P95_ERA5": float(np.percentile(era, 95)),

        "P95_OBS": float(np.percentile(obs, 95)),

        "P99_ERA5": float(np.percentile(era, 99)),

        "P99_OBS": float(np.percentile(obs, 99))

    }

print()
print("=" * 60)
print("Treino concluído.")
print(f"Modelos criados: {len(modelos)}")
print("=" * 60)

# ==========================================================
# GUARDAR MODELOS
# ==========================================================

ficheiro_modelo = os.path.join(
    PASTA_SAIDA,
    "Modelos_QM_Radiacao.pkl"
)

print("\nGuardar modelos...")

with open(ficheiro_modelo, "wb") as f:
    pickle.dump(
        modelos,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("Modelos guardados com sucesso.")

# ==========================================================
# CRIAR RESUMO DAS ESTAÇÕES
# ==========================================================

resumo = []

for nome, modelo in modelos.items():

    resumo.append({

        "Estacao": nome,

        "Codigo": modelo["Codigo"],

        "Latitude": modelo["Latitude"],

        "Longitude": modelo["Longitude"],

        "N": modelo["N"],

        "Media_ERA5": modelo["Media_ERA5"],

        "Media_OBS": modelo["Media_OBS"],

        "Mediana_ERA5": modelo["Mediana_ERA5"],

        "Mediana_OBS": modelo["Mediana_OBS"],

        "Std_ERA5": modelo["Std_ERA5"],

        "Std_OBS": modelo["Std_OBS"],

        "Min_ERA5": modelo["Min_ERA5"],

        "Min_OBS": modelo["Min_OBS"],

        "Max_ERA5": modelo["Max_ERA5"],

        "Max_OBS": modelo["Max_OBS"],

        "P5_ERA5": modelo["P5_ERA5"],

        "P5_OBS": modelo["P5_OBS"],

        "P25_ERA5": modelo["P25_ERA5"],

        "P25_OBS": modelo["P25_OBS"],

        "P50_ERA5": modelo["P50_ERA5"],

        "P50_OBS": modelo["P50_OBS"],

        "P75_ERA5": modelo["P75_ERA5"],

        "P75_OBS": modelo["P75_OBS"],

        "P95_ERA5": modelo["P95_ERA5"],

        "P95_OBS": modelo["P95_OBS"],

        "P99_ERA5": modelo["P99_ERA5"],

        "P99_OBS": modelo["P99_OBS"]

    })

resumo = pd.DataFrame(resumo)

ficheiro_csv = os.path.join(
    PASTA_SAIDA,
    "Resumo_Modelos_QM_Radiacao.csv"
)

resumo.to_csv(
    ficheiro_csv,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# ESTATÍSTICAS FINAIS
# ==========================================================

print("\n" + "=" * 60)
print("EMPIRICAL QUANTILE MAPPING - RADIAÇÃO TREINADO")
print("=" * 60)

print(f"Número de estações : {len(modelos)}")
print(f"Total observações  : {len(df):,}")
print("Variável           : Radiação Solar Mensal (MJ/m²)")

print("\nFicheiros criados:")
print(f"Modelo : {ficheiro_modelo}")
print(f"Resumo : {ficheiro_csv}")

print("\nConcluído.")
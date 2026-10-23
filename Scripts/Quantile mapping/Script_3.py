# ============================================================
# SCRIPT 3 - Aplicação Quantile Mapping + IDW à Radiação Solar
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# ============================================================
# 1. CAMINHOS
# ============================================================

ERA5_FILE = r""

STATIONS_FILE = r""

QM_SINGLE_FILE = r""

OUTPUT_FILE = r""

# Número de estações usadas por pixel
N_NEIGHBOURS = 5

# ============================================================
# 2. CARREGAR ERA5
# ============================================================

print("A carregar ERA5...")

era5 = xr.open_dataset(ERA5_FILE)

# Variável correta de radiação no ERA5 (ssrd_mj_mensal)
VAR = "ssrd_mj_mensal"

radiacao = era5[VAR].sortby("valid_time")

print("ERA5 carregado:", radiacao.shape)

# ============================================================
# 3. CARREGAR MODELOS QM
# ============================================================

print("\nA carregar modelos QM...")

if not os.path.exists(QM_SINGLE_FILE):
    raise FileNotFoundError(QM_SINGLE_FILE)

with open(QM_SINGLE_FILE, "rb") as f:
    qm_models = pickle.load(f)

print(f"Modelos carregados: {len(qm_models)}")

# ============================================================
# 4. CARREGAR ESTAÇÕES
# ============================================================

stations = pd.read_csv(STATIONS_FILE)

stations.columns = (
    stations.columns
    .str.strip()
    .str.lower()
)

stations = stations.dropna(
    subset=["latitude", "longitude"]
)

if "estacao_csv" in stations.columns:
    id_col = "estacao_csv"
elif "estacao_snirh" in stations.columns:
    id_col = "estacao_snirh"
elif "codigo" in stations.columns:
    id_col = "codigo"
else:
    id_col = stations.columns[0]

stations["id"] = (
    stations[id_col]
    .astype(str)
    .str.upper()
    .str.strip()
)

stations = stations[
    stations["id"].isin(qm_models.keys())
]

station_ids = stations["id"].values
station_lat = stations["latitude"].values
station_lon = stations["longitude"].values

print(f"Estações disponíveis: {len(stations)}")

if len(stations) == 0:
    raise RuntimeError(
        "Nenhuma estação do CSV coincide com o ficheiro de modelos."
    )

# ============================================================
# 5. KD TREE
# ============================================================

coords = np.column_stack([
    station_lat,
    station_lon
])

tree = cKDTree(coords)

# ============================================================
# 6. FUNÇÕES AUXILIARES
# ============================================================

def obter_estacoes_idw(lat, lon, n=N_NEIGHBOURS):
    n_k = min(n, len(station_ids))

    distancia, indices = tree.query(
        [lat, lon],
        k=n_k
    )

    if n_k == 1:
        distancia = np.array([distancia])
        indices = np.array([indices])

    distancia = np.maximum(
        distancia,
        1e-6
    )

    pesos = 1.0 / (distancia ** 2)
    pesos = pesos / pesos.sum()

    ids = station_ids[indices]
    return ids, pesos


# ============================================================
# EMPIRICAL QUANTILE MAPPING (COM PROTEÇÃO DE LIMITES FÍSICOS)
# ============================================================

def aplicar_qm(modelo, valores):
    valores = np.asarray(
        valores,
        dtype=np.float32
    )

    era = modelo["ERA5"]
    obs = modelo["OBS"]
    cdf_era = modelo["CDF_ERA5"]
    cdf_obs = modelo["CDF_OBS"]

    prob = np.interp(
        valores,
        era,
        cdf_era,
        left=0.0,
        right=1.0
    )

    corrigido = np.interp(
        prob,
        cdf_obs,
        obs,
        left=obs[0],
        right=obs[-1]
    )

    # 1. Garantir limite físico inferior (não pode ser negativo)
    corrigido = np.maximum(
        corrigido,
        0.0
    )

    # 2. Tecto físico superior estrito (impede valores fantasma > 1000 MJ/m²)
    corrigido = np.minimum(
        corrigido,
        1000.0
    )

    return corrigido


# ============================================================
# 7. PREPARAR GRELHA DE SAÍDA
# ============================================================

print("\nA preparar grelha corrigida...")

latitudes = radiacao.latitude.values
longitudes = radiacao.longitude.values
tempos = radiacao.valid_time.values

radiacao_corrigida = np.full(
    radiacao.shape,
    np.nan,
    dtype=np.float32
)

total_pixels = len(latitudes) * len(longitudes)
contador = 0

print(f"Dimensão ERA5: {radiacao.shape}")
print("A aplicar Quantile Mapping + IDW...")

# ============================================================
# 8. PROCESSAR TODOS OS PIXELS
# ============================================================

for i, lat in enumerate(latitudes):
    for j, lon in enumerate(longitudes):
        contador += 1

        if contador % 200 == 0 or contador == total_pixels:
            print(
                f"Processados {contador}/{total_pixels} pixels..."
            )

        ids_estacoes, pesos = obter_estacoes_idw(
            lat,
            lon
        )

        serie_pixel = radiacao[:, i, j].values

        # Ignorar pixels sem dados
        if np.all(np.isnan(serie_pixel)):
            continue

        correcoes = []
        pesos_utilizados = []

        for sid, peso in zip(ids_estacoes, pesos):
            if sid not in qm_models:
                continue

            modelo = qm_models[sid]

            serie_corrigida = aplicar_qm(
                modelo,
                serie_pixel
            )

            correcoes.append(
                serie_corrigida
            )

            pesos_utilizados.append(
                peso
            )

        if len(correcoes) == 0:
            continue

        correcoes = np.asarray(
            correcoes,
            dtype=np.float32
        )

        pesos_utilizados = np.asarray(
            pesos_utilizados,
            dtype=np.float32
        )

        pesos_utilizados /= pesos_utilizados.sum()

        valor_final = np.average(
            correcoes,
            axis=0,
            weights=pesos_utilizados
        )

        radiacao_corrigida[:, i, j] = np.maximum(
            valor_final,
            0.0
        )

print("\nCorreção concluída.")

# ============================================================
# 9. GUARDAR NETCDF
# ============================================================

print("\nA criar NetCDF final...")

ds_corrigido = xr.Dataset(
    {
        "ssrd_mj_mensal": (
            ["valid_time", "latitude", "longitude"],
            radiacao_corrigida
        )
    },
    coords={
        "valid_time": tempos,
        "latitude": latitudes,
        "longitude": longitudes,
    }
)

ds_corrigido["ssrd_mj_mensal"].attrs = {
    "long_name": "Radiação Solar Líquida Mensal corrigida por Empirical Quantile Mapping + IDW",
    "units": "MJ/m^2",
    "method": "Empirical Quantile Mapping + IDW",
    "stations_used": f"{N_NEIGHBOURS} estações mais próximas"
}

print("\nA guardar ficheiro...")

ds_corrigido.to_netcdf(
    OUTPUT_FILE
)

# ============================================================
# 10. ESTATÍSTICAS FINAIS
# ============================================================

print("\n=======================================")
print("PROCESSO CONCLUÍDO")
print("=======================================")

print(ds_corrigido)

print()
print(
    "Média ERA5:",
    float(np.nanmean(radiacao.values))
)
print(
    "Média Corrigida:",
    float(np.nanmean(radiacao_corrigida))
)

print()
print(
    "Desvio padrão ERA5:",
    float(np.nanstd(radiacao.values))
)
print(
    "Desvio padrão Corrigido:",
    float(np.nanstd(radiacao_corrigida))
)

print()
print(
    "Mínimo ERA5:",
    float(np.nanmin(radiacao.values))
)
print(
    "Mínimo Corrigido:",
    float(np.nanmin(radiacao_corrigida))
)

print()
print(
    "Máximo ERA5:",
    float(np.nanmax(radiacao.values))
)
print(
    "Máximo Corrigido:",
    float(np.nanmax(radiacao_corrigida))
)

print()
print("NetCDF guardado em:")
print(OUTPUT_FILE)

print("\nTudo concluído com sucesso.")
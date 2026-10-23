import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from tqdm import tqdm

# ==========================================================
# CAMINHOS
# ==========================================================

ERA5_NC = r""

COORD_CSV = r""

PASTA_ESTACOES = r""

PASTA_SAIDA = r""

os.makedirs(PASTA_SAIDA, exist_ok=True)

# ==========================================================
# LER ERA5
# ==========================================================

print("A abrir ERA5...")

ds = xr.open_dataset(ERA5_NC)

latitudes = ds.latitude.values
longitudes = ds.longitude.values

lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

coords_grid = np.column_stack([
    lat_grid.ravel(),
    lon_grid.ravel()
])

tree = cKDTree(coords_grid)

# Converter valid_time do ERA5 para o primeiro dia de cada mês
tempo = pd.to_datetime(ds.valid_time.values).to_period("M").to_timestamp()

print("ERA5 carregado.")

# ==========================================================
# LER COORDENADAS
# ==========================================================

coord = pd.read_csv(COORD_CSV)

# Normalizar os nomes das colunas
coord.columns = coord.columns.str.strip().str.lower()

colunas_possiveis = ["estacao_csv", "estacao_snirh", "estacao", "nome"]
coluna_estacao = None

for col in colunas_possiveis:
  if col in coord.columns:
    coluna_estacao = col
    break

if coluna_estacao is None:
  raise KeyError("Não foi encontrada nenhuma coluna identificadora da estação no CSV de coordenadas!")

print(f"A utilizar a coluna '{coluna_estacao}' como identificador da estação.")

coord["estacao_csv"] = coord[coluna_estacao].astype(str).str.upper().str.strip()

coord["latitude"] = pd.to_numeric(coord["latitude"], errors="coerce")
coord["longitude"] = pd.to_numeric(coord["longitude"], errors="coerce")

coord = coord.dropna(subset=["latitude", "longitude"])

R = 6371

def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ==========================================================

todos = []
relatorio = []

# Procurar os ficheiros de radiação convertidos
ficheiros = sorted(
    glob.glob(
        os.path.join(
            PASTA_ESTACOES,
            "*_MJ_convertido.csv"
        )
    )
)

print(f"\nForam encontrados {len(ficheiros)} ficheiros de radiação.\n")

# ==========================================================

for ficheiro in tqdm(ficheiros):
    estacao = "DESCONHECIDA"
    try:
        nome = os.path.basename(ficheiro)

        # Extração exata do nome da estação removendo os sufixos possíveis
        estacao = nome.replace(
            "_Radiacao_diaria_radiacao_mensal_MJ_convertido.csv",
            ""
        ).replace(
            "_radiacao_diaria_radiacao_mensal_MJ_convertido.csv",
            ""
        ).replace(
            "_MJ_convertido.csv",
            ""
        ).upper().strip()

        if estacao not in coord["estacao_csv"].values:
            relatorio.append({
                "Estacao": estacao,
                "Estado": "Sem coordenadas"
            })
            continue

        linha = coord.loc[coord["estacao_csv"] == estacao].iloc[0]

        lat = float(linha["latitude"])
        lon = float(linha["longitude"])

        if not np.isfinite(lat) or not np.isfinite(lon):
            relatorio.append({
                "Estacao": estacao,
                "Estado": "Coordenadas inválidas"
            })
            continue

        codigo = linha["codigo"] if "codigo" in linha else ""

        # ==================================================
        # Pixel ERA5 mais próximo
        # ==================================================

        _, indice = tree.query([lat, lon])
        i, j = np.unravel_index(indice, lat_grid.shape)

        lat_pixel = latitudes[i]
        lon_pixel = longitudes[j]

        distancia = haversine(lat, lon, lat_pixel, lon_pixel)

        # Série ERA5 corrigida para a radiação mensal (ssrd_mj_mensal)
        serie_era5 = ds["ssrd_mj_mensal"].isel(latitude=i, longitude=j).values

        era5 = pd.DataFrame({
            "valid_time": tempo,
            "ERA5": serie_era5
        })

        # ==================================================
        # Ler observações mensais da estação
        # ==================================================

        obs = pd.read_csv(ficheiro)

        if len(obs) == 0:
            relatorio.append({
                "Estacao": estacao,
                "Estado": "CSV vazio"
            })
            continue

        # Normalizar nomes das colunas para minúsculas
        obs.columns = obs.columns.str.strip().str.lower()

        if "ano" not in obs.columns or "mes" not in obs.columns:
            relatorio.append({
                "Estacao": estacao,
                "Estado": "Colunas ano/mes não encontradas"
            })
            continue

        # Garantir conversão correta de Ano e Mes para data mensal fixa (dia 1)
        obs["valid_time"] = pd.to_datetime(
            obs["ano"].astype(str) + "-" + obs["mes"].astype(str).str.zfill(2) + "-01",
            errors="coerce"
        )

        # Identificar dinamicamente a coluna de valor de radiação
        col_valor = [c for c in obs.columns if c not in ["ano", "mes", "valid_time"]][0]

        obs["OBS"] = pd.to_numeric(obs[col_valor].astype(str).str.replace(",", "."), errors="coerce")
        obs = obs.dropna(subset=["valid_time", "OBS"])
        obs = obs[["valid_time", "OBS"]]

        # ==================================================
        # Juntar ERA5 + Observações
        # ==================================================

        tabela = pd.merge(
            era5,
            obs,
            on="valid_time",
            how="inner"
        )

        tabela = tabela.dropna()

        if len(tabela) == 0:
            relatorio.append({
                "Estacao": estacao,
                "Estado": "Sem datas em comum"
            })
            continue

        tabela["Codigo"] = codigo
        tabela["Estacao"] = estacao
        tabela["Latitude"] = lat
        tabela["Longitude"] = lon
        tabela["Distancia_km"] = round(distancia, 2)

        todos.append(tabela)

        relatorio.append({
            "Estacao": estacao,
            "Meses": len(tabela),
            "Distancia_km": round(distancia, 2),
            "Estado": "OK"
        })

    except Exception as e:
        relatorio.append({
            "Estacao": estacao,
            "Estado": str(e)
        })

# ==========================================================
print("\nA juntar resultados...")
df_relatorio = pd.DataFrame(relatorio)
print(df_relatorio)

if len(todos) == 0:
    raise RuntimeError("Nenhuma estação foi processada com sucesso. Verifique o relatório acima.")

treino = pd.concat(todos, ignore_index=True)

treino.to_parquet(
    os.path.join(PASTA_SAIDA, "Treino_QM_Radiacao.parquet"),
    index=False
)

treino.to_csv(
    os.path.join(PASTA_SAIDA, "Treino_QM_Radiacao.csv"),
    index=False
)

df_relatorio.to_csv(
    os.path.join(PASTA_SAIDA, "Relatorio_Estacoes_Radiacao.csv"),
    index=False
)

print("\n========================================")
print("PROCESSO TERMINADO")
print("========================================")

print(f"\nTotal de linhas: {len(treino):,}")
print(f"Estações processadas com sucesso: {treino.Estacao.nunique()}")
print(f"Resultados guardados em: {PASTA_SAIDA}")
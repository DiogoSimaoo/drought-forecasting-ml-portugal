import joblib
import pandas as pd
import xarray as xr
import numpy as np
import os

# 1. Carregar os componentes guardados
scaler = joblib.load(r"")
model = joblib.load(r"")
feature_names_treino = joblib.load(r"")

# 2. Carregar o NetCDF
input_nc = r""
ds = xr.open_dataset(input_nc)

spei1 = ds["SPEI_1"].values
spei3 = ds["SPEI_3"].values
spei6 = ds["SPEI_6"].values
spei12 = ds["SPEI_12"].values
spi6 = ds["SPI_6"].values

time = ds["time"].values
lat = ds["latitude"].values
lon = ds["longitude"].values

LOOKBACK = 36

# 3. Funções auxiliares integradas para engenharia de caraterísticas
def valor_seguro(array, indice):
    valor = array[indice]
    if np.isnan(valor) or np.isinf(valor):
        return 0.0
    return float(valor)

def estatisticas_janela(array, t, meses):
    inicio = max(0, t - meses)
    janela = array[inicio:t]
    janela = janela[np.isfinite(janela)]
    if len(janela) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(np.mean(janela)), float(np.std(janela)), float(np.min(janela)), float(np.max(janela))]

def criar_features(s1, s3, s6, s12, spi, t):
    if t < LOOKBACK:
        return None
    features = []
    nomes = []

    atuais = [
        ("SPEI1_t", s1[t]), ("SPEI3_t", s3[t]), ("SPEI6_t", s6[t]), ("SPEI12_t", s12[t]), ("SPI6_t", spi[t])
    ]
    for nome, valor in atuais:
        arr_ref = s1 if nome == "SPEI1_t" else s3 if nome == "SPEI3_t" else s6 if nome == "SPEI6_t" else s12 if nome == "SPEI12_t" else spi
        features.append(valor_seguro(arr_ref, t))
        nomes.append(nome)

    for lag in [1, 2, 3, 6]:
        features.append(valor_seguro(s1, t - lag))
        nomes.append(f"SPEI1_lag{lag}")
    for lag in [1, 2, 3, 6, 12]:
        features.append(valor_seguro(s3, t - lag))
        nomes.append(f"SPEI3_lag{lag}")
    for lag in [1, 2, 3, 6, 9, 12]:
        features.append(valor_seguro(s6, t - lag))
        nomes.append(f"SPEI6_lag{lag}")
    for lag in [1, 3, 6, 12, 18]:
        features.append(valor_seguro(s12, t - lag))
        nomes.append(f"SPEI12_lag{lag}")
    for lag in [1, 2, 3, 6, 12]:
        features.append(valor_seguro(spi, t - lag))
        nomes.append(f"SPI6_lag{lag}")

    features.append(valor_seguro(s1, t) - valor_seguro(s1, t - 1)); nomes.append("SPEI1_change_1")
    features.append(valor_seguro(s1, t) - valor_seguro(s1, t - 3)); nomes.append("SPEI1_change_3")
    features.append(valor_seguro(s3, t) - valor_seguro(s3, t - 1)); nomes.append("SPEI3_change_1")
    features.append(valor_seguro(s3, t) - valor_seguro(s3, t - 3)); nomes.append("SPEI3_change_3")
    features.append(valor_seguro(s3, t) - valor_seguro(s3, t - 6)); nomes.append("SPEI3_change_6")
    features.append(valor_seguro(s6, t) - valor_seguro(s6, t - 1))
    nomes.append("SPEI6_change_1")
    features.append(valor_seguro(s6, t) - valor_seguro(s6, t - 3))
    nomes.append("SPEI6_change_3")
    features.append(valor_seguro(s6, t) - valor_seguro(s6, t - 6))
    nomes.append("SPEI6_change_6")
    features.append(valor_seguro(s6, t) - valor_seguro(s6, t - 12))
    nomes.append("SPEI6_change_12")
    features.append(valor_seguro(s12, t) - valor_seguro(s12, t - 6)); nomes.append("SPEI12_change_6")
    features.append(valor_seguro(s12, t) - valor_seguro(s12, t - 12)); nomes.append("SPEI12_change_12")

    for meses in [3, 6, 12]:
        media, std, minimo, maximo = estatisticas_janela(s1, t, meses)
        features.extend([media, std, minimo, maximo])
        nomes.extend([f"SPEI1_mean_{meses}", f"SPEI1_std_{meses}", f"SPEI1_min_{meses}", f"SPEI1_max_{meses}"])
    for meses in [3, 6, 12]:
        media, std, minimo, maximo = estatisticas_janela(s3, t, meses)
        features.extend([media, std, minimo, maximo])
        nomes.extend([f"SPEI3_mean_{meses}", f"SPEI3_std_{meses}", f"SPEI3_min_{meses}", f"SPEI3_max_{meses}"])
    for meses in [6, 12, 24, 36]:
        media, std, minimo, maximo = estatisticas_janela(s6, t, meses)
        features.extend([media, std, minimo, maximo])
        nomes.extend([f"SPEI6_mean_{meses}", f"SPEI6_std_{meses}", f"SPEI6_min_{meses}", f"SPEI6_max_{meses}"])
    for meses in [12, 24, 36]:
        media, std, minimo, maximo = estatisticas_janela(s12, t, meses)
        features.extend([media, std, minimo, maximo])
        nomes.extend([f"SPEI12_mean_{meses}", f"SPEI12_std_{meses}", f"SPEI12_min_{meses}", f"SPEI12_max_{meses}"])
    for meses in [3, 6, 12]:
        media, std, minimo, maximo = estatisticas_janela(spi, t, meses)
        features.extend([media, std, minimo, maximo])
        nomes.extend([f"SPI6_mean_{meses}", f"SPI6_std_{meses}", f"SPI6_min_{meses}", f"SPI6_max_{meses}"])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array(features, dtype=np.float32), nomes

# Inicializar matriz de resultados com dimensões completas (preenchida com NaN para períodos sem lookback)
n_time = len(time)
n_lat = len(lat)
n_lon = len(lon)

array_previsoes = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
array_probabilidades = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

mapa_inverso = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}

print("A processar grelha espacial e temporal para previsão...")
for i, llat in enumerate(lat):
    for j, llon in enumerate(lon):
        s1 = spei1[:, i, j]
        s3 = spei3[:, i, j]
        s6 = spei6[:, i, j]
        s12 = spei12[:, i, j]
        spi = spi6[:, i, j]

        if np.isnan(s6).all():
            continue

        X_pixel = []
        indices_t = []

        for t in range(LOOKBACK, n_time):
            resultado = criar_features(s1, s3, s6, s12, spi, t)
            if resultado is None:
                continue
            feat, nomes = resultado
            dict_feat = dict(zip(nomes, feat))
            feat_ordenado = [dict_feat.get(col, 0.0) for col in feature_names_treino]
            X_pixel.append(feat_ordenado)
            indices_t.append(t)

        if len(X_pixel) > 0:
            X_pixel = np.array(X_pixel, dtype=np.float32)
            dados_escalados = scaler.transform(X_pixel)
            
            # Previsão e cálculo da confiança máxima (probabilidade associada à classe prevista)
            preds = model.predict(dados_escalados)
            probs = model.predict_proba(dados_escalados)
            max_probs = np.max(probs, axis=1)

            classes_reais = [mapa_inverso[val] for val in preds]

            for idx_t, t_val in enumerate(indices_t):
                array_previsoes[t_val, i, j] = classes_reais[idx_t]
                array_probabilidades[t_val, i, j] = max_probs[idx_t]

# 4. Criar Dataset em Xarray para exportação NetCDF4
ds_out = xr.Dataset(
    {
        "drought_class_pred": (["time", "latitude", "longitude"], array_previsoes),
        "prediction_confidence": (["time", "latitude", "longitude"], array_probabilidades)
    },
    coords={
        "time": time,
        "latitude": lat,
        "longitude": lon
    }
)

# Atribuir metadados úteis
ds_out["drought_class_pred"].attrs = {
    "description": "Previsão das classes de seca (Random Forest)",
    "mapping": "-3: Extrema, -2: Severa, -1: Moderada, 0: Normal, 1: Ligeiramente húmida, 2: Moderadamente húmida, 3: Muito húmida"
}
ds_out["prediction_confidence"].attrs = {
    "description": "Probabilidade/Confiança associada à classe prevista pelo Random Forest (0 a 1)"
}

# 5. Guardar o ficheiro NetCDF de saída
output_nc_path = r"C:\Users\isa125585\Desktop\modelo_tentar\previsoes_seca_output.nc"
ds_out.to_netcdf(output_nc_path)

print(f"Ficheiro NetCDF gerado com sucesso em: {output_nc_path}")
ds.close()
ds_out.close()
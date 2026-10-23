import os
import json
import joblib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error
)

from imblearn.over_sampling import SMOTE

# ======================================================
# PARÂMETROS
# ======================================================

LOOKBACK = 36
input_nc = r""

output_dir = r""

os.makedirs(output_dir, exist_ok=True)

output_modelo = os.path.join(
    output_dir,
    "modelo_rf.pkl"
)

output_scaler = os.path.join(
    output_dir,
    "scaler.pkl"
)

output_metrics = os.path.join(
    output_dir,
    "metricas.txt"
)

output_cm = os.path.join(
    output_dir,
    "matriz_confusao.png"
)

output_importance = os.path.join(
    output_dir,
    "importancia_features.png"
)


# ======================================================
# CLASSES SPEI (7 CLASSES)
# ======================================================

def classificar_spei(val):

    c = np.full_like(
        val,
        np.nan,
        dtype=np.float32
    )

    # --------------------------------------------------
    # SECA
    # --------------------------------------------------

    c[val <= -2.0] = -3

    c[
        (val > -2.0) &
        (val <= -1.5)
    ] = -2

    c[
        (val > -1.5) &
        (val < -0.99)
    ] = -1

    # --------------------------------------------------
    # NORMAL
    # --------------------------------------------------

    c[
        (val >= -0.99) &
        (val <= 0.99)
    ] = 0

    # --------------------------------------------------
    # HÚMIDO
    # --------------------------------------------------

    c[
        (val > 0.99) &
        (val < 1.5)
    ] = 1

    c[
        (val >= 1.5) &
        (val < 2.0)
    ] = 2

    c[val >= 2.0] = 3

    return c


# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================

def valor_seguro(array, indice):

    valor = array[indice]

    if np.isnan(valor) or np.isinf(valor):
        return 0.0

    return float(valor)


def estatisticas_janela(array, t, meses):

    inicio = max(0, t - meses)

    janela = array[inicio:t]

    janela = janela[
        np.isfinite(janela)
    ]

    if len(janela) == 0:

        return [
            0.0,
            0.0,
            0.0,
            0.0
        ]

    return [

        float(np.mean(janela)),

        float(np.std(janela)),

        float(np.min(janela)),

        float(np.max(janela))
    ]


# ======================================================
# CRIAR FEATURES (COM SPEI-1)
# ======================================================

def criar_features(
    s1,
    s3,
    s6,
    s12,
    spi,
    t
):

    if t < LOOKBACK:

        return None

    features = []
    nomes = []

    # ==================================================
    # VALORES ATUAIS
    # ==================================================

    atuais = [
        ("SPEI1_t", s1[t]),
        ("SPEI3_t", s3[t]),
        ("SPEI6_t", s6[t]),
        ("SPEI12_t", s12[t]),
        ("SPI6_t", spi[t])
    ]

    for nome, valor in atuais:

        arr_ref = (
            s1 if nome == "SPEI1_t"
            else s3 if nome == "SPEI3_t"
            else s6 if nome == "SPEI6_t"
            else s12 if nome == "SPEI12_t"
            else spi
        )

        features.append(
            valor_seguro(arr_ref, t)
        )

        nomes.append(nome)

    # ==================================================
    # LAGS
    # ==================================================

    # --------------------------------------------------
    # SPEI-1
    # --------------------------------------------------

    for lag in [1, 2, 3, 6]:

        features.append(
            valor_seguro(s1, t - lag)
        )

        nomes.append(
            f"SPEI1_lag{lag}"
        )

    # --------------------------------------------------
    # SPEI-3
    # --------------------------------------------------

    for lag in [1, 2, 3, 6, 12]:

        features.append(
            valor_seguro(s3, t - lag)
        )

        nomes.append(
            f"SPEI3_lag{lag}"
        )

    # --------------------------------------------------
    # SPEI-6
    # --------------------------------------------------

    for lag in [1, 2, 3, 6, 9, 12]:

        features.append(
            valor_seguro(s6, t - lag)
        )

        nomes.append(
            f"SPEI6_lag{lag}"
        )

    # --------------------------------------------------
    # SPEI-12
    # --------------------------------------------------

    for lag in [1, 3, 6, 12, 18]:

        features.append(
            valor_seguro(s12, t - lag)
        )

        nomes.append(
            f"SPEI12_lag{lag}"
        )

    # --------------------------------------------------
    # SPI-6
    # --------------------------------------------------

    for lag in [1, 2, 3, 6, 12]:

        features.append(
            valor_seguro(spi, t - lag)
        )

        nomes.append(
            f"SPI6_lag{lag}"
        )

    # ==================================================
    # TENDÊNCIAS
    # ==================================================

    # SPEI-1
    features.append(valor_seguro(s1, t) - valor_seguro(s1, t - 1))
    nomes.append("SPEI1_change_1")

    features.append(valor_seguro(s1, t) - valor_seguro(s1, t - 3))
    nomes.append("SPEI1_change_3")

    # SPEI-3
    features.append(valor_seguro(s3, t) - valor_seguro(s3, t - 1))
    nomes.append("SPEI3_change_1")

    features.append(valor_seguro(s3, t) - valor_seguro(s3, t - 3))
    nomes.append("SPEI3_change_3")

    features.append(valor_seguro(s3, t) - valor_seguro(s3, t - 6))
    nomes.append("SPEI3_change_6")

    # SPEI-6
    features.append(valor_seguro(s6, t) - valor_seguro(s6, t - 1))
    nomes.append("SPEI6_change_1")

    features.append(valor_seguro(s6, t) - valor_seguro(s6, t - 3))
    nomes.append("SPEI6_change_3")

    features.append(valor_seguro(s6, t) - valor_seguro(s6, t - 6))
    nomes.append("SPEI6_change_6")

    features.append(valor_seguro(s6, t) - valor_seguro(s6, t - 12))
    nomes.append("SPEI6_change_12")

    # SPEI-12
    features.append(valor_seguro(s12, t) - valor_seguro(s12, t - 6))
    nomes.append("SPEI12_change_6")

    features.append(valor_seguro(s12, t) - valor_seguro(s12, t - 12))
    nomes.append("SPEI12_change_12")

    # ==================================================
    # ESTATÍSTICAS SPEI-1
    # ==================================================

    for meses in [3, 6, 12]:

        media, std, minimo, maximo = estatisticas_janela(s1, t, meses)

        features.extend([media, std, minimo, maximo])

        nomes.extend([
            f"SPEI1_mean_{meses}",
            f"SPEI1_std_{meses}",
            f"SPEI1_min_{meses}",
            f"SPEI1_max_{meses}"
        ])

    # ==================================================
    # ESTATÍSTICAS SPEI-3
    # ==================================================

    for meses in [3, 6, 12]:

        media, std, minimo, maximo = estatisticas_janela(s3, t, meses)

        features.extend([media, std, minimo, maximo])

        nomes.extend([
            f"SPEI3_mean_{meses}",
            f"SPEI3_std_{meses}",
            f"SPEI3_min_{meses}",
            f"SPEI3_max_{meses}"
        ])

    # ==================================================
    # ESTATÍSTICAS SPEI-6
    # ==================================================

    for meses in [6, 12, 24, 36]:

        media, std, minimo, maximo = estatisticas_janela(s6, t, meses)

        features.extend([media, std, minimo, maximo])

        nomes.extend([
            f"SPEI6_mean_{meses}",
            f"SPEI6_std_{meses}",
            f"SPEI6_min_{meses}",
            f"SPEI6_max_{meses}"
        ])

    # ==================================================
    # ESTATÍSTICAS SPEI-12
    # ==================================================

    for meses in [12, 24, 36]:

        media, std, minimo, maximo = estatisticas_janela(s12, t, meses)

        features.extend([media, std, minimo, maximo])

        nomes.extend([
            f"SPEI12_mean_{meses}",
            f"SPEI12_std_{meses}",
            f"SPEI12_min_{meses}",
            f"SPEI12_max_{meses}"
        ])

    # ==================================================
    # SPI-6
    # ==================================================

    for meses in [3, 6, 12]:

        media, std, minimo, maximo = estatisticas_janela(spi, t, meses)

        features.extend([media, std, minimo, maximo])

        nomes.extend([
            f"SPI6_mean_{meses}",
            f"SPI6_std_{meses}",
            f"SPI6_min_{meses}",
            f"SPI6_max_{meses}"
        ])

    # ==================================================
    # LIMPEZA
    # ==================================================

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return (
        np.array(features, dtype=np.float32),
        nomes
    )


# ======================================================
# MAIN
# ======================================================

def main():

    print("==============================================")
    print("A abrir NetCDF...")
    print("==============================================")

    ds = xr.open_dataset(input_nc)

    # ==================================================
    # VARIÁVEIS (Incluindo SPEI_1)
    # ==================================================

    spei1 = ds["SPEI_1"].values
    spei3 = ds["SPEI_3"].values
    spei6 = ds["SPEI_6"].values
    spei12 = ds["SPEI_12"].values
    spi6 = ds["SPI_6"].values

    time = ds["time"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    print(f"Meses: {len(time)}")
    print(f"Latitudes: {len(lat)}")
    print(f"Longitudes: {len(lon)}")

    # ==================================================
    # CLASSIFICAÇÃO
    # ==================================================

    print("\nA classificar SPEI-6...")
    classe = classificar_spei(spei6)

    # ==================================================
    # DATASET
    # ==================================================

    X = []
    y = []
    feature_names = None

    print("\n==============================================")
    print("A criar dataset t -> t+1...")
    print("==============================================")

    for i in range(len(lat)):
        print(f"Latitude {i+1}/{len(lat)}")

        for j in range(len(lon)):

            s1 = spei1[:, i, j]
            s3 = spei3[:, i, j]
            s6 = spei6[:, i, j]
            s12 = spei12[:, i, j]
            spi = spi6[:, i, j]

            if np.isnan(s6).all():
                continue

            for t in range(LOOKBACK, len(time) - 1):

                resultado = criar_features(
                    s1,
                    s3,
                    s6,
                    s12,
                    spi,
                    t
                )

                if resultado is None:
                    continue

                feat, nomes = resultado

                if feature_names is None:
                    feature_names = nomes

                target = classe[t + 1, i, j]

                if np.isnan(target) or np.isinf(target):
                    continue

                X.append(feat)
                y.append(target)

    # ==================================================
    # NUMPY E RESTO DO PIPELINE
    # ==================================================

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print("\n==============================================")
    print("DATASET FINAL")
    print("==============================================")
    print(f"X = {X.shape}")
    print(f"y = {y.shape}")
    print(f"Número de features = {X.shape[1]}")

    # Distribuição de classes
    nomes_classes = {
        -3: "Seca extrema",
        -2: "Seca severa",
        -1: "Seca moderada",
         0: "Normal",
         1: "Moderadamente húmido",
         2: "Muito húmido",
         3: "Extremamente húmido"
    }

    classes_originais, contagens = np.unique(y, return_counts=True)
    for c, n in zip(classes_originais, contagens):
        print(f"{int(c):2d} | {nomes_classes[int(c)]:22s} | {n}")

    mapa = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
    y_enc = np.array([mapa[int(v)] for v in y], dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    print(f"Treino: {X_train.shape}")
    print(f"Teste: {X_test.shape}")

    print("\nA aplicar StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nA aplicar SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Treino após SMOTE: {X_train.shape}")

    print("\nA treinar Random Forest...")
    modelo = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    modelo.fit(X_train, y_train)
    print("Modelo treinado.")

    y_pred = modelo.predict(X_test)

    labels = list(range(7))
    nomes_classes_lista = [
        "Seca extrema", "Seca severa", "Seca moderada",
        "Normal", "Moderadamente húmido", "Muito húmido", "Extremamente húmido"
    ]

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=nomes_classes_lista, digits=4, zero_division=0
    )

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    mae_classes = mean_absolute_error(y_test, y_pred)

    print(f"\nAccuracy:      {accuracy:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    print(f"MAE classes:   {mae_classes:.4f}")

    # Guardar métricas e artefactos
    with open(output_metrics, "w", encoding="utf-8") as f:
        f.write("MODELO RANDOM FOREST - SPEI t+1\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Macro F1: {macro_f1:.6f}\n")
        f.write(str(cm))
        f.write("\n\n" + report)

    joblib.dump(modelo, output_modelo)
    joblib.dump(scaler, output_scaler)
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.pkl"))

    print("\n✅ PROCESSO TERMINADO")
    ds.close()

if __name__ == "__main__":
    main()
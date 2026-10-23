# ==========================================================
# ET0 FAO56 MENSAL COM PyETo
# PARTE 1/4
# ==========================================================

import os
import numpy as np
import pandas as pd
import xarray as xr
import pyeto

# ==========================================================
# CAMINHOS
# ==========================================================

DIR_BASE = r"C:\Users\isa125585\Desktop\ficheiros_nao_corrigidos"

FILE_TEMP = os.path.join(DIR_BASE, "temperatura_algarve_1950_2024_FINAL.nc")
FILE_HR = os.path.join(DIR_BASE, "humidade_relativa.nc")
FILE_VENTO = os.path.join(DIR_BASE, "vento_media_mensal_1950_2024.nc")
FILE_RAD = os.path.join(DIR_BASE, "era5_ssrd_mensal_1950_2024.nc")

OUT_FILE = os.path.join(DIR_BASE, "ET0_FAO56_mensal.nc")

# ==========================================================
# ALTITUDE MÉDIA (m)
# ==========================================================

ALTITUDE = 0.0

# ==========================================================
# LER DADOS
# ==========================================================

print("A abrir ficheiros...")

temp = xr.open_dataset(FILE_TEMP)
hr = xr.open_dataset(FILE_HR)
vento = xr.open_dataset(FILE_VENTO)
rad = xr.open_dataset(FILE_RAD)

# ==========================================================
# RENOMEAR valid_time
# ==========================================================

if "valid_time" in vento.coords:
    vento = vento.rename({"valid_time": "time"})

if "valid_time" in rad.coords:
    rad = rad.rename({"valid_time": "time"})

# ==========================================================
# COLOCAR TODAS AS DATAS NO PRIMEIRO DIA DO MÊS
# ==========================================================

temp = temp.assign_coords(
    time=temp.time.values.astype("datetime64[M]")
)

hr = hr.assign_coords(
    time=hr.time.values.astype("datetime64[M]")
)

vento = vento.assign_coords(
    time=vento.time.values.astype("datetime64[M]")
)

rad = rad.assign_coords(
    time=rad.time.values.astype("datetime64[M]")
)

# ==========================================================
# INTERPOLAR RADIAÇÃO PARA A GRELHA DA TEMPERATURA
# ==========================================================

print("Interpolando radiação...")

rad = rad.interp(
    latitude=temp.latitude,
    longitude=temp.longitude
)

# ==========================================================
# ALINHAR DATASETS
# ==========================================================

temp, hr, vento, rad = xr.align(
    temp,
    hr,
    vento,
    rad,
    join="inner"
)

print(temp)

# ==========================================================
# VARIÁVEIS
# ==========================================================

Tmean = temp["temp_media_C"]

Tmax = temp["temp_max_C"]

Tmin = temp["temp_min_C"]

RH = hr["rh"]

u2 = vento["u2"]

Rs_month = rad["ssrd_mj_mensal"]

# ==========================================================
# NÚMERO DE DIAS POR MÊS
# ==========================================================

days = xr.DataArray(
    temp.time.dt.days_in_month.values,
    coords={"time": temp.time},
    dims="time"
)

# ==========================================================
# CONVERTER RADIAÇÃO
# MJ m-2 mês-1 -> MJ m-2 dia-1
# ==========================================================

Rs = Rs_month / days

print("\nResumo dos dados:")
print(f"Meses: {len(temp.time)}")
print(f"Latitudes: {len(temp.latitude)}")
print(f"Longitudes: {len(temp.longitude)}")
print(f"Período: {str(temp.time.values[0])[:10]} até {str(temp.time.values[-1])[:10]}")

# ==========================================================
# PARTE 2/4
# CÁLCULO DAS CONSTANTES MENSAIS FAO56
# ==========================================================

lat_rad = pyeto.deg2rad(temp.latitude.values)

pressure = pyeto.atm_pressure(ALTITUDE)

psy = pyeto.psy_const(pressure)

times = pd.to_datetime(temp.time.values)

Ra_list = []
Rso_list = []

print("A calcular Ra e Rso...")

for t in times:

    J = pd.Timestamp(t.year, t.month, 15).dayofyear

    dr = pyeto.inv_rel_dist_earth_sun(J)

    dec = pyeto.sol_dec(J)

    ws = np.array([
        pyeto.sunset_hour_angle(phi, dec)
        for phi in lat_rad
    ])

    Ra = np.array([
        pyeto.et_rad(phi, dec, ws_i, dr)
        for phi, ws_i in zip(lat_rad, ws)
    ])

    Rso = pyeto.cs_rad(
        ALTITUDE,
        Ra
    )

    Ra_list.append(Ra)

    Rso_list.append(Rso)

Ra = xr.DataArray(

    np.stack(Ra_list),

    coords={
        "time": temp.time,
        "latitude": temp.latitude
    },

    dims=("time", "latitude")

)

Rso = xr.DataArray(

    np.stack(Rso_list),

    coords={
        "time": temp.time,
        "latitude": temp.latitude
    },

    dims=("time", "latitude")

)

Ra = Ra.broadcast_like(Tmean)

Rso = Rso.broadcast_like(Tmean)

# ==========================================================
# RADIAÇÃO LÍQUIDA
# ==========================================================

# ==========================================================
# RADIAÇÃO LÍQUIDA DE ONDA CURTA
# ==========================================================

Rns = (1.0 - 0.23) * Rs

# ==========================================================
# PRESSÃO DE VAPOR SATURADO
# ==========================================================

es_tmax = 0.6108 * np.exp((17.27 * Tmax) / (Tmax + 237.3))

es_tmin = 0.6108 * np.exp((17.27 * Tmin) / (Tmin + 237.3))

es = (es_tmax + es_tmin) / 2.0

# ==========================================================
# PRESSÃO DE VAPOR REAL
# ==========================================================

ea = es * RH / 100.0

# ==========================================================
# DECLIVE DA CURVA DE PRESSÃO DE VAPOR
# ==========================================================

Delta = (
    4098
    * 0.6108
    * np.exp((17.27 * Tmean) / (Tmean + 237.3))
    / ((Tmean + 237.3) ** 2)
)
# ==========================================================
# PARTE 3/4
# RADIAÇÃO LÍQUIDA E ET0
# ==========================================================

print("A calcular radiação líquida...")

# Temperaturas em Kelvin
TmaxK = pyeto.celsius2kelvin(Tmax)
TminK = pyeto.celsius2kelvin(Tmin)

# ==========================================================
# RADIAÇÃO LÍQUIDA DE ONDA LONGA
# ==========================================================

# ==========================================================
# RADIAÇÃO LÍQUIDA DE ONDA LONGA (FAO-56)
# ==========================================================

SIGMA = 4.903e-9

# Evitar divisões por zero
Rs_Rso = xr.where(Rso > 0, Rs / Rso, 0)

# Limitar ao intervalo recomendado pela FAO
Rs_Rso = Rs_Rso.clip(min=0.0, max=1.0)

Rnl = (
    SIGMA
    * ((TmaxK**4 + TminK**4) / 2.0)
    * (0.34 - 0.14 * np.sqrt(ea))
    * (1.35 * Rs_Rso - 0.35)
)

# ==========================================================
# RADIAÇÃO LÍQUIDA TOTAL
# ==========================================================

Rn = Rns - Rnl
# ==========================================================
# FLUXO DE CALOR NO SOLO
# ==========================================================

# Para valores mensais assume-se G = 0
G = 0.0

# ==========================================================
# ET0 (mm/dia)
# ==========================================================

ET0_day = pyeto.fao56_penman_monteith(
    Rn,
    Tmean + 273.15,
    u2,
    es,
    ea,
    Delta,
    psy,
    G
)

# Eliminar valores negativos

ET0_day = xr.where(
    ET0_day < 0,
    0,
    ET0_day
)

# ==========================================================
# CONVERTER PARA mm/mês
# ==========================================================

ET0 = ET0_day * days

ET0.name = "ET0"

ET0.attrs["long_name"] = (
    "Evapotranspiração de referência FAO-56"
)

ET0.attrs["units"] = "mm mes-1"

print("ET0 calculado.")
# ==========================================================
# PARTE 4/4
# GUARDAR RESULTADOS
# ==========================================================

print("\n===================================")
print("Resumo do ET0")
print("===================================")

print("ET0 mínimo :", float(ET0.min().values))
print("ET0 máximo :", float(ET0.max().values))
print("ET0 médio  :", float(ET0.mean().values))

print("Número de NaN :", int(np.isnan(ET0.values).sum()))

# ==========================================================
# CRIAR DATASET
# ==========================================================

ds_out = xr.Dataset(
    {
        "ET0": ET0
    }
)

ds_out["ET0"].attrs = {
    "long_name": "Evapotranspiracao de referencia FAO-56 Penman-Monteith",
    "units": "mm mes-1",
    "standard_name": "reference_evapotranspiration"
}

ds_out.attrs = {
    "title": "ET0 mensal calculado pelo método FAO-56 Penman-Monteith",
    "source": "ERA5 corrigido por Quantile Mapping + IDW",
    "software": "Python + PyETo + xarray",
    "Conventions": "CF-1.8"
}

# ==========================================================
# COMPRESSÃO
# ==========================================================

encoding = {
    "ET0": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(np.nan)
    }
}

# ==========================================================
# GUARDAR NETCDF
# ==========================================================

print("\nA guardar ficheiro...")

ds_out.to_netcdf(
    OUT_FILE,
    encoding=encoding
)

print("\n===================================")
print("PROCESSO TERMINADO")
print("===================================")

print(f"Ficheiro criado:\n{OUT_FILE}")

print("\nResumo do Dataset:")
print(ds_out)

print("\nPrimeiro mês:")
print(ds_out["ET0"].isel(time=0))

print("\nÚltimo mês:")
print(ds_out["ET0"].isel(time=-1))
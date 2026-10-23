import os
import subprocess
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import requests
from tqdm import tqdm

# ==========================================
# CONFIGURAÇÃO
# ==========================================

LINKS_FILE = "links.txt"

TIMEOUT = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://snirh.apambiente.pt/",
    "Origin": "https://snirh.apambiente.pt"
}

session = requests.Session()
session.headers.update(HEADERS)

hoje = datetime.today().strftime("%d/%m/%Y")


# ==========================================
# CORRIGIR URL
# ==========================================

def corrigir_url(url):

    url = url.replace("http://", "https://")
    url = url.replace("snirh.pt", "snirh.apambiente.pt")
    url = url.replace("30/12/1899", "01/01/1910")
    url = url.replace("dd/mm/aaaa", hoje)

    return url


# ==========================================
# DOWNLOAD COM REQUESTS
# ==========================================

def download_requests(url, destino):

    try:

        r = session.get(url, timeout=TIMEOUT)

        if r.status_code != 200:
            return False

        with open(destino, "wb") as f:
            f.write(r.content)

        if os.path.getsize(destino) < 100:
            return False

        return True

    except:

        return False


# ==========================================
# DOWNLOAD COM CURL
# ==========================================

def download_curl(url, destino):

    comando = [
        "curl",
        "-L",
        "--compressed",
        "--retry",
        "5",
        "-A",
        HEADERS["User-Agent"],
        "-H",
        "Referer: https://snirh.apambiente.pt/",
        "-H",
        "Origin: https://snirh.apambiente.pt",
        "-o",
        destino,
        url,
    ]

    try:

        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
        )

        if resultado.returncode != 0:
            return False

        if not os.path.exists(destino):
            return False

        if os.path.getsize(destino) < 100:
            return False

        return True

    except:

        return False
# ==========================================
# LER links.txt
# ==========================================

if not os.path.exists(LINKS_FILE):
    print(f"Erro: não encontrei '{LINKS_FILE}'")
    exit()

with open(LINKS_FILE, "r", encoding="utf-8") as f:
    linhas = f.readlines()

# Remover cabeçalho
linhas = linhas[1:]

erros = []
sucessos = 0
ignorados = 0

print(f"Foram encontradas {len(linhas)} entradas.\n")

# ==========================================
# CICLO PRINCIPAL
# ==========================================

for linha in tqdm(linhas, desc="Downloads"):

    linha = linha.strip()

    if linha == "":
        continue

    partes = linha.split(",")

    if len(partes) < 3:
        continue

    estacao = partes[0].strip()
    variavel = partes[1].strip()
    url = corrigir_url(partes[2].strip())

    # criar pasta da variável
    os.makedirs(variavel, exist_ok=True)

    ficheiro = os.path.join(
        variavel,
        f"{estacao}_{variavel}.csv"
    )

    # ignorar se já existe
    if os.path.exists(ficheiro):

        ignorados += 1
        continue

    # ======================================
    # Primeiro tenta requests
    # ======================================

    ok = download_requests(url, ficheiro)

    # ======================================
    # Se falhar tenta curl
    # ======================================

    if not ok:

        ok = download_curl(url, ficheiro)

    # ======================================
    # Resultado
    # ======================================

    if ok:

        sucessos += 1

    else:

        erros.append({
            "estacao": estacao,
            "variavel": variavel,
            "url": url
        })
# ==========================================
# RESUMO
# ==========================================

print("\n")

print("=" * 60)
print("DOWNLOAD TERMINADO")
print("=" * 60)

print(f"Downloads efetuados : {sucessos}")
print(f"Já existentes       : {ignorados}")
print(f"Falhas              : {len(erros)}")

# ==========================================
# GUARDAR ERROS
# ==========================================

if len(erros) > 0:

    with open("erros.txt","w",encoding="utf-8") as f:

        f.write("ESTACAO;VARIAVEL;URL\n")

        for e in erros:

            f.write(
                f"{e['estacao']};{e['variavel']};{e['url']}\n"
            )

    print("\nFoi criado o ficheiro erros.txt")

else:

    print("\nTodos os downloads foram concluídos com sucesso.")

# ==========================================
# APAGAR CSV VAZIOS
# ==========================================

print("\nA verificar ficheiros...")

apagados = 0

for pasta in os.listdir("."):

    if not os.path.isdir(pasta):
        continue

    for ficheiro in os.listdir(pasta):

        if not ficheiro.lower().endswith(".csv"):
            continue

        caminho = os.path.join(pasta, ficheiro)

        try:

            if os.path.getsize(caminho) < 100:

                os.remove(caminho)
                apagados += 1

        except:
            pass

print(f"Ficheiros inválidos removidos: {apagados}")

print("\nConcluído.")
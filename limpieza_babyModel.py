# -*- coding: utf-8 -*-
"""
limpieza_babyModel.py

ES: Limpia el CSV crudo de TikTok y genera el dataset que usa el modelo.
EN: Cleans the raw TikTok CSV and builds the dataset used by the model.

Uso / Usage:
    python limpieza_babyModel.py
"""

# =============================================================================
# CONFIGURACIÓN / CONFIGURATION
# =============================================================================

# Entrada / Input
ARCHIVO_ENTRADA = "datos_sucios/dataset_13oct26.csv"
SEPARADOR = None            # None = autodetectar / autodetect; "\t" = TSV
CODIFICACION = "utf-8-sig"  # "utf-8-sig" = UTF-8 con BOM, para Excel / UTF-8 with BOM, for Excel

# Salidas / Outputs
CARPETA_SALIDA = "datos_limpios"
# Parquet: para el modelo; conserva listas y enteros tal cual.
# Parquet: for the model; keeps lists and ints as-is.
SALIDA_PARQUET = "dataset_babyModel.parquet"
# CSV: para revisar a mano / Excel; las listas van como texto JSON.
# CSV: for manual review / Excel; lists are stored as JSON text.
SALIDA_CSV = "dataset_babyModel.csv"
ARCHIVO_LOG = "limpieza_log.txt"

# Columnas crudas / Raw columns
COL_ID = "video_id" 
COL_TIPO = "tipo"
COL_GRUPO = "grupo"
COL_URL = "url"
COL_AUTOR = "autor"
COL_CAPTION = "caption"
COL_TEXTO_VIDEO = "texto_en_video"
COL_EMOJIS_VIDEO = "emojis_en_video"
COL_COMENTARIOS_TOTAL = "comentarios_total"
COL_COMENTARIOS = "comentarios"
COL_ERROR = "error"

# Claves del JSON de comentarios / Comment JSON keys
COM_USUARIO = "usuario"
COM_TEXTO = "comentario"
COM_LIKES = "likes"

# Tipo de contenido -> binario / Content type -> binary
MAPA_TIPO = {"foto": 0, "fotos": 0, "photo": 0, "video": 1}

# Marcadores / Markers
SEP_COMENTARIOS = " [SEP] "     # entre comentarios / between comments
MARCA_AUTOR = "[AUTOR] "        # comentario escrito por el creador / comment by the creator
TOKEN_USUARIO = "@user"         # reemplaza cualquier @mención / replaces any @mention
SEP_OCR = " | "                 # entre líneas del OCR / between OCR lines

# También anonimizar @menciones en la caption / Also anonymize @mentions in the caption
ANONIMIZAR_MENCIONES_CAPTION = True

# Códigos que se quitan de cada emoji (selectores de variante y tonos de piel)
# Code points removed from each emoji (variation selectors and skin tones)
CODIGOS_A_QUITAR = {0xFE0E, 0xFE0F} | set(range(0x1F3FB, 0x1F400))

# Columnas finales, en orden / Final columns, in order
COLUMNAS_SALIDA = [
    "video_id", "autor", "grupo", "grupo_doble", "grupo_segundo",
    "tipo", "caption", "texto_en_video", "emojis_en_video",
    "comentarios_total", "comentarios_total_imputado", "n_comentarios_capturados",
    "comentarios", "datos_parciales",
]

# =============================================================================
# CÓDIGO / CODE
# =============================================================================

#librerías / libraries
import json
import os
import re
import unicodedata
import emoji
import pandas as pd

_MENCION = re.compile(r"@[\w.]+")
_ESPACIOS = re.compile(r"\s+")
_log = []

# FUNCIONES AUXILIARES / AUXILIARY FUNCTIONS

#Recibe un mensaje, lo guarda en la lista _log y lo imprime / Receives a message, saves it in the _log list and prints it
def log(msg):
    # Guardar y mostrar aviso / Store and print warning
    _log.append(msg)
    print(msg)

# Normaliza texto a NFC y quita espacios al inicio y final / Normalizes text to NFC and strips leading/trailing spaces
def texto(x):
    # NaN -> "" ; normaliza a NFC (conserva acentos y ñ) / NaN -> "" ; NFC keeps accents and ñ
    if not isinstance(x, str):
        return ""
    return unicodedata.normalize("NFC", x).strip() # NFD quita acentos y ñ / NFD removes accents and ñ


def emoji_a_token(e):
    # 🇲🇽 -> "U+1F1F2_U+1F1FD" ; 👏🏻 -> "U+1F44F" ; 4️⃣ -> "U+0034_U+20E3"
    codigos = [f"U+{ord(c):04X}" for c in e if ord(c) not in CODIGOS_A_QUITAR]
    return "_".join(codigos)

#recibe un string, reemplaza cada emoji por su token y devuelve el string resultante
#receives a string, replaces each emoji with its token and returns the resulting string
def emojis_a_unicode(s):
    # Reemplaza cada emoji por su token, separado por espacios
    # Replace each emoji with its token, space-separated
    partes, pos = [], 0
    for m in emoji.emoji_list(s):
        partes.append(s[pos:m["match_start"]])
        tok = emoji_a_token(m["emoji"])
        if tok:
            partes.append(f" {tok} ")
        pos = m["match_end"]
    partes.append(s[pos:])
    return _ESPACIOS.sub(" ", "".join(partes)).strip()


def lista_emojis(s):
    # String de emojis -> lista de tokens / Emoji string -> list of tokens
    return [t for t in (emoji_a_token(m["emoji"]) for m in emoji.emoji_list(texto(s))) if t]


def limpiar_caption(s):
    s = texto(s)
    if ANONIMIZAR_MENCIONES_CAPTION:
        s = _MENCION.sub(TOKEN_USUARIO, s)
    return emojis_a_unicode(s)

#ocr: optcal character recognition (reconocimiento óptico de caracteres)
def limpiar_ocr(s, autor):
    # Quita [?], marca de agua "TikTok", @autor y líneas repetidas (sin importar mayúsculas)
    # Remove [?], "TikTok" watermark, @author and repeated lines (case-insensitive)
    vistas, salida = set(), []
    for linea in texto(s).split("|"):
        linea = re.sub(r"\[\?\]", "", linea).strip()
        clave = linea.lower()
        if not linea or clave == "tiktok" or (autor and clave == f"@{autor}".lower()):
            continue
        if clave in vistas:
            continue
        vistas.add(clave)
        salida.append(emojis_a_unicode(linea))
    return SEP_OCR.join(salida)
#nota [?]: el OCR de TikTok a veces incluye "[?]" en lugar de un caracter que no reconoce. Se quita para no confundir al modelo.


def limpiar_comentarios(s, autor, vid):
    # JSON -> (string gigante, n capturados) / JSON -> (really long string, n captured)
    s = texto(s)
    if not s:
        return "", 0
    try:
        datos = json.loads(s)
        if not isinstance(datos, list):
            raise ValueError("no es lista / not a list")
    except (json.JSONDecodeError, ValueError) as e:
        log(f"[JSON] {vid}: comentarios mal formados, se tratan como vacíos / malformed, treated as empty ({e})")
        return "", 0

    datos = [c for c in datos if isinstance(c, dict) and texto(c.get(COM_TEXTO))]
    # Ordenar por likes, mayor a menor / Sort by likes, descending
    datos.sort(key=lambda c: pd.to_numeric(c.get(COM_LIKES), errors="coerce") or 0, reverse=True)

    salida = []
    for c in datos:
        t = emojis_a_unicode(_MENCION.sub(TOKEN_USUARIO, texto(c.get(COM_TEXTO))))
        # Marcar comentarios del creador / Flag comments by the creator
        if autor and texto(c.get(COM_USUARIO)).lower() == autor.lower():
            t = MARCA_AUTOR + t
        salida.append(t)
    return SEP_COMENTARIOS.join(salida), len(salida)


def parsear_grupo(valor):
    # "3,4" -> (3, 4) ; "2" -> (2, None) ; vacío / empty -> None
    nums = [int(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", texto(valor))]
    nums = [n for n in nums if 0 <= n <= 4]
    if not nums:
        return None
    return nums[0], (nums[1] if len(nums) > 1 else None), len(nums)


def tipo_binario(tipo, url):
    # foto=0, video=1
    t = MAPA_TIPO.get(texto(tipo).lower())
    if t is not None:
        return t
    # Si no hay tipo, se deduce de la URL / If no type, infer from URL
    u = texto(url).lower()
    if "/photo/" in u:
        return 0
    if "/video/" in u:
        return 1
    return None


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    df = pd.read_csv(ARCHIVO_ENTRADA, sep=SEPARADOR, engine="python", encoding=CODIFICACION, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    log(f"Filas leídas / Rows read: {len(df)}")

    # --- Duplicados: conservar la primera / Duplicates: keep first ---
    df[COL_ID] = df[COL_ID].map(texto)
    for vid, g in df[df.duplicated(COL_ID, keep=False)].groupby(COL_ID):
        etiquetas = g[COL_GRUPO].map(texto).unique().tolist()
        if len(etiquetas) > 1:
            log(f"[DUP] {vid}: etiquetas distintas / different labels {etiquetas}; se conserva la primera / keeping first")
    n0 = len(df)
    df = df.drop_duplicates(COL_ID, keep="first")
    log(f"Duplicados eliminados / Duplicates removed: {n0 - len(df)}")

    filas, sin_grupo, sin_tipo, sin_texto = [], 0, 0, 0
    for _, r in df.iterrows():
        vid, autor = r[COL_ID], texto(r.get(COL_AUTOR))

        # Etiqueta: sin grupo -> fuera / Label: no group -> drop
        g = parsear_grupo(r.get(COL_GRUPO))
        if g is None:
            sin_grupo += 1
            continue
        grupo, grupo2, n_etiquetas = g
        if n_etiquetas > 2:
            log(f"[GRUPO] {vid}: más de 2 etiquetas / more than 2 labels '{r.get(COL_GRUPO)}'; se usan las 2 primeras / using first 2")

        # Tipo: sin forma de deducirlo -> fuera / Type: impossible to infer -> drop
        tipo = tipo_binario(r.get(COL_TIPO), r.get(COL_URL))
        if tipo is None:
            sin_tipo += 1
            log(f"[TIPO] {vid}: tipo desconocido y URL sin /photo/ ni /video/ / unknown type; descartada / dropped")
            continue

        caption = limpiar_caption(r.get(COL_CAPTION))
        ocr = limpiar_ocr(r.get(COL_TEXTO_VIDEO), autor)
        comentarios, n_cap = limpiar_comentarios(r.get(COL_COMENTARIOS), autor, vid)

        # Sin nada de texto -> fuera / No text at all -> drop
        if not caption and not ocr and not comentarios:
            sin_texto += 1
            continue

        # Total de comentarios / Total comments
        total = pd.to_numeric(texto(r.get(COL_COMENTARIOS_TOTAL)).replace(",", ""), errors="coerce")
        imputado = int(pd.isna(total))
        if imputado:
            # Vacío -> se usan los capturados / Missing -> use captured count
            total = n_cap
        # TikTok a veces reporta menos de los capturados (desfase al contar); se usa el mayor.
        # TikTok sometimes reports fewer than captured (count lag); use the larger one.
        total = int(max(total, n_cap))

        filas.append({
            "video_id": vid,
            "autor": autor,
            "grupo": grupo,
            "grupo_doble": int(grupo2 is not None),
            "grupo_segundo": grupo2,
            "tipo": tipo,
            "caption": caption,
            "texto_en_video": ocr,
            "emojis_en_video": lista_emojis(r.get(COL_EMOJIS_VIDEO)),
            "comentarios_total": total,
            "comentarios_total_imputado": imputado,
            "n_comentarios_capturados": n_cap,
            "comentarios": comentarios,
            # Hubo error al capturar / Capture error occurred
            "datos_parciales": int(bool(texto(r.get(COL_ERROR)))),
        })

    out = pd.DataFrame(filas, columns=COLUMNAS_SALIDA)
    # Enteros que admiten vacío / Nullable ints
    out["grupo_segundo"] = out["grupo_segundo"].astype("Int64")

    log(f"Descartadas sin grupo / Dropped, no label: {sin_grupo}")
    log(f"Descartadas sin tipo / Dropped, no type: {sin_tipo}")
    log(f"Descartadas sin texto / Dropped, no text: {sin_texto}")
    log(f"Filas finales / Final rows: {len(out)}")
    if len(out):
        log(f"Grupos / Groups: {out['grupo'].value_counts().sort_index().to_dict()}")
        log(f"Dobles / Double-labeled: {out['grupo_doble'].sum()}  |  "
            f"Parciales / Partial: {out['datos_parciales'].sum()}  |  "
            f"Total imputado / Imputed total: {out['comentarios_total_imputado'].sum()}")

    # Parquet para el modelo / Parquet for the model
    out.to_parquet(os.path.join(CARPETA_SALIDA, SALIDA_PARQUET), index=False)
    # CSV para revisar; listas como JSON / CSV for review; lists as JSON
    csv = out.copy()
    csv["emojis_en_video"] = csv["emojis_en_video"].map(json.dumps)
    csv.to_csv(os.path.join(CARPETA_SALIDA, SALIDA_CSV), index=False, encoding="utf-8-sig")

    with open(os.path.join(CARPETA_SALIDA, ARCHIVO_LOG), "w", encoding="utf-8") as f:
        f.write("\n".join(_log))
    print(f"\nListo / Done -> {CARPETA_SALIDA}/")


if __name__ == "__main__":
    main()

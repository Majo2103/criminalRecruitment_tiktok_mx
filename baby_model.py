# -*- coding: utf-8 -*-
"""
baby_model.py

ES: Primer modelo (texto + imagen) para clasificar posts en grupos 0-4.
    Objetivo: máximo recall del grupo 0 y del subconjunto {3,4}.
EN: First model (text + image) to classify posts into groups 0-4.
    Goal: maximize recall for group 0 and for the {3,4} subset.

Uso / Usage:
    python baby_model.py
"""

# =============================================================================
# CONFIGURACIÓN / CONFIGURATION
# =============================================================================

# --- Rutas / Paths ---
RUTA_PARQUET = "datos_limpios/dataset_babyModel.parquet"
CARPETA_IMAGENES = "datos_sucios/audiovisuales_12oct26/imagenes"   # imagenes/<video_id>/*.jpg
CARPETA_VIDEOS = "datos_sucios/audiovisuales_12oct26/videos"       # videos/<video_id>.mp4
CARPETA_SALIDA = "resultados_babyModel"
EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".webp")

# --- Muestra / Sample ---
SEMILLA = 210303
PROP_TEST = 0.30
AGRUPAR_POR_AUTOR = True      # ninguna cuenta en train y test a la vez / no account in both
N_FOLDS_CV = 5                # CV dentro de train para elegir umbrales / CV in train for thresholds

# --- Imágenes / Images ---
N_IMAGENES = 1                # n del main: 1 = foto a la mitad del video / middle frame
MODELO_IMAGEN = "openai/clip-vit-base-patch32"   # se descarga la 1a vez / downloaded on first run
USAR_EMBEDDING_IMAGEN = True  # vector de 512 números de CLIP / 512-d CLIP vector
USAR_SONDAS_IMAGEN = True     # parecido a frases (abajo) / similarity to prompts (below)
GUARDAR_FRAMES = False        # guarda las imágenes usadas para revisarlas / save frames to inspect
# Sondas: frases en inglés (CLIP entiende mejor inglés) / Prompts in English (CLIP works best)
SONDAS_IMAGEN = {
    "arma": "a photo of a gun or a rifle",
    "arma_mano": "a person holding a weapon",
    "drogas": "a photo of illegal drugs",
    "dinero": "a pile of cash money",
    "blindado": "an armored military truck",
    "pelea": "people fighting or hitting someone",
    "uniforme": "soldiers or police in tactical uniform",
    "texto": "a screenshot or image with a lot of text",
    "baile": "people dancing",
    "auto": "a car or a pickup truck",
    "animal": "a cute pet animal",
    "musica": "a singer or a music band performing",
}

# --- Texto / Text ---
TFIDF_MIN_DF = 2
TFIDF_WORD_MAX = 20000
TFIDF_CHAR_MAX = 30000

# --- Variables numéricas / Numeric features ---
# datos_parciales queda fuera: es un error de captura, no contenido.
# datos_parciales is left out: it's a capture error, not content.
USAR_DATOS_PARCIALES = False

# --- Clasificador / Classifier ---
LOGREG_C = 1.0

# --- Objetivo y umbrales / Goal and thresholds ---
CLASES = [0, 1, 2, 3, 4]
RECLUT = {3, 4}
OBJETIVO = "media"            # "media", "minimo" o "armonica" de recall_0 y recall_34
PISO_RECALL_1 = 0.25          # que no desaparezcan 1 y 2 / keep groups 1 and 2 alive
PISO_RECALL_2 = 0.25
REJILLA = [round(0.05 + 0.025 * i, 3) for i in range(37)]   # 0.05 ... 0.95

# =============================================================================
# CÓDIGO / CODE
# =============================================================================
import os
import re
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# 1. Imágenes / Images
# -----------------------------------------------------------------------------
def _frames_video(ruta, n):
    # Frames en los momentos (i/(n+1))*t, i=1..n / Frames at times (i/(n+1))*t
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = []
    for i in range(1, n + 1):
        frac = i / (n + 1)
        ok = False
        if total > 0:
            # Saltar al frame / Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * total))
            ok, img = cap.read()
        if not ok and fps > 0 and total > 0:
            # Respaldo por milisegundos / Fallback by milliseconds
            cap.set(cv2.CAP_PROP_POS_MSEC, frac * total / fps * 1000)
            ok, img = cap.read()
        if ok:
            # OpenCV usa BGR; PIL usa RGB / OpenCV is BGR; PIL is RGB
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def get_images(n, id_video):
    """
    ES: Hasta n imágenes del post. Fotos: las primeras n de la carpeta. Video: n frames repartidos.
    EN: Up to n images. Photos: first n in the folder. Video: n evenly spaced frames.
    """
    id_video = str(id_video)
    carpeta = os.path.join(CARPETA_IMAGENES, id_video)
    if os.path.isdir(carpeta):
        # Post de fotos / Photo post
        archivos = sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES_IMAGEN))
        imagenes = []
        for f in archivos[:n]:
            try:
                imagenes.append(Image.open(os.path.join(carpeta, f)).convert("RGB"))
            except OSError:
                print(f"  ! imagen ilegible / unreadable image: {id_video}/{f}")
        return imagenes
    video = os.path.join(CARPETA_VIDEOS, f"{id_video}.mp4")
    if os.path.isfile(video):
        # Post de video / Video post
        return _frames_video(video, n)
    return []   # sin archivos / no files


class CodificadorImagen:
    """ES: Convierte imágenes en vectores con CLIP. / EN: Turns images into vectors with CLIP."""

    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(MODELO_IMAGEN).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(MODELO_IMAGEN)
        print(f"  CLIP cargado en / loaded on: {self.device}")
        self.sondas = self._textos(list(SONDAS_IMAGEN.values())) if USAR_SONDAS_IMAGEN else None

    @staticmethod
    def _vector(salida, clave):
        # Compatibilidad entre versiones de transformers / Compat across transformers versions
        if hasattr(salida, clave):
            salida = getattr(salida, clave)
        elif hasattr(salida, "pooler_output"):
            salida = salida.pooler_output
        return salida / salida.norm(dim=-1, keepdim=True)

    def _textos(self, textos):
        with self.torch.no_grad():
            x = self.proc(text=textos, return_tensors="pt", padding=True).to(self.device)
            return self._vector(self.model.get_text_features(**x), "text_embeds").cpu().numpy()

    def imagenes(self, imgs):
        # Promedio normalizado de las n imágenes / Normalized mean of the n images
        with self.torch.no_grad():
            x = self.proc(images=imgs, return_tensors="pt").to(self.device)
            v = self._vector(self.model.get_image_features(**x), "image_embeds").cpu().numpy()
        v = v.mean(0)
        return v / np.linalg.norm(v)


def variables_imagen(ids, n):
    """
    ES: Embedding + sondas por post, con caché en disco (CLIP es lento).
    EN: Embedding + probes per post, cached on disk (CLIP is slow).
    """
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    nombre = re.sub(r"\W", "_", MODELO_IMAGEN)
    ruta_cache = os.path.join(CARPETA_SALIDA, f"cache_imagen_{nombre}_n{n}.joblib")
    cache = joblib.load(ruta_cache) if os.path.exists(ruta_cache) else {}
    faltan = [i for i in ids if i not in cache]
    print(f"Imágenes / Images: {len(ids) - len(faltan)} en caché, {len(faltan)} por procesar")

    if faltan:
        cod = CodificadorImagen()
        for k, vid in enumerate(faltan, 1):
            imgs = get_images(n, vid)
            if GUARDAR_FRAMES:
                os.makedirs(os.path.join(CARPETA_SALIDA, "frames"), exist_ok=True)
                for j, im in enumerate(imgs, 1):
                    im.save(os.path.join(CARPETA_SALIDA, "frames", f"{vid}_{j}.jpg"))
            if imgs:
                emb = cod.imagenes(imgs)
                sond = emb @ cod.sondas.T if cod.sondas is not None else None
                cache[vid] = {"emb": emb, "sondas": sond, "n": len(imgs)}
            else:
                cache[vid] = {"emb": None, "sondas": None, "n": 0}
            if k % 50 == 0 or k == len(faltan):
                print(f"  {k}/{len(faltan)}")
                joblib.dump(cache, ruta_cache)   # guardar avance / save progress

    # Armar tabla; sin imagen -> NaN (luego se imputa) / Build table; no image -> NaN (imputed later)
    dim = next((c["emb"].shape[0] for c in cache.values() if c["emb"] is not None), 512)
    filas = []
    for vid in ids:
        c = cache[vid]
        fila = {"sin_imagen": int(c["emb"] is None)}
        if USAR_EMBEDDING_IMAGEN:
            emb = c["emb"] if c["emb"] is not None else np.full(dim, np.nan)
            fila.update({f"img_{j}": v for j, v in enumerate(emb)})
        if USAR_SONDAS_IMAGEN:
            s = c["sondas"] if c["sondas"] is not None else np.full(len(SONDAS_IMAGEN), np.nan)
            fila.update({f"sonda_{k}": v for k, v in zip(SONDAS_IMAGEN, s)})
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    print(f"  Posts sin imagen / Posts without image: {tabla['sin_imagen'].sum()}")
    return tabla


# -----------------------------------------------------------------------------
# 2. Texto / Text
# -----------------------------------------------------------------------------
_TOKEN = re.compile(r"U\+[0-9A-F]+(?:_U\+[0-9A-F]+)*|\[AUTOR\]|@user|\w+")
_SEGMENTOS = re.compile(r"\s\[SEP\]\s|\s\|\s|\[CAP\]|\[OCR\]|\[EMO\]")
_MARCAS = re.compile(r"U\+[0-9A-F]+(?:_U\+[0-9A-F]+)*|\[SEP\]|\[AUTOR\]|\[CAP\]|\[OCR\]|\[EMO\]|@user")


def analizador_palabras(texto):
    """
    ES: Palabras + pares, sin cruzar de un comentario/línea a otro. Emojis U+... = 1 token.
    EN: Words + bigrams, never across comments/lines. Emoji U+... = 1 token.
    """
    salida = []
    for seg in _SEGMENTOS.split(texto):
        toks = [t if t.startswith(("U+", "[")) else t.lower() for t in _TOKEN.findall(seg)]
        salida += toks
        salida += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return salida


def preprocesador_char(texto):
    # Para n-gramas de caracteres: sin códigos de emoji ni marcas / No emoji codes or markers
    return _MARCAS.sub(" ", texto).lower()


def construir_tabla(df, img):
    # Une texto, numéricas e imagen en una tabla / Joins text, numeric and image into one table
    X = pd.DataFrame(index=df.index)
    emojis = df["emojis_en_video"].map(lambda l: " ".join(l) if l is not None else "")
    X["txt_contenido"] = ("[CAP] " + df["caption"].fillna("") +
                          " [OCR] " + df["texto_en_video"].fillna("") +
                          " [EMO] " + emojis)
    X["txt_coment"] = df["comentarios"].fillna("")
    X["tipo"] = df["tipo"].astype(int)
    X["log_com_total"] = np.log1p(df["comentarios_total"].astype(float))
    X["log_com_capt"] = np.log1p(df["n_comentarios_capturados"].astype(float))
    X["com_imputado"] = df["comentarios_total_imputado"].astype(int)
    X["n_autor"] = df["comentarios"].fillna("").str.count(r"\[AUTOR\]")
    if USAR_DATOS_PARCIALES:
        X["datos_parciales"] = df["datos_parciales"].astype(int)
    return pd.concat([X.reset_index(drop=True), img.reset_index(drop=True)], axis=1)


def crear_modelo(X):
    # Pipeline: transformaciones por bloque + regresión logística / per-block transforms + LR
    numericas = [c for c in ["tipo", "log_com_total", "log_com_capt", "com_imputado", "n_autor",
                             "datos_parciales", "sin_imagen"] if c in X.columns]
    img_cols = [c for c in X.columns if c.startswith(("img_", "sonda_"))]
    bloques = []
    for nombre, col in [("cont", "txt_contenido"), ("com", "txt_coment")]:
        bloques += [
            (f"{nombre}_pal", TfidfVectorizer(analyzer=analizador_palabras, min_df=TFIDF_MIN_DF,
                                              max_features=TFIDF_WORD_MAX, sublinear_tf=True), col),
            (f"{nombre}_char", TfidfVectorizer(analyzer="char_wb", preprocessor=preprocesador_char,
                                               ngram_range=(2, 5), min_df=TFIDF_MIN_DF,
                                               max_features=TFIDF_CHAR_MAX, sublinear_tf=True), col),
        ]
    bloques.append(("num", StandardScaler(), numericas))
    if img_cols:
        bloques.append(("img", Pipeline([("imp", SimpleImputer(strategy="mean")),
                                         ("esc", StandardScaler())]), img_cols))
    return Pipeline([
        ("pre", ColumnTransformer(bloques, sparse_threshold=1.0)),
        ("clf", LogisticRegression(C=LOGREG_C, max_iter=5000, class_weight="balanced")),
    ])


def probabilidades(modelo, X):
    # Matriz n x 5 aunque falte alguna clase / n x 5 matrix even if a class is missing
    P = np.zeros((len(X), len(CLASES)))
    p = modelo.predict_proba(X)
    for j, c in enumerate(modelo.classes_):
        P[:, CLASES.index(c)] = p[:, j]
    return P


# -----------------------------------------------------------------------------
# 3. Decisión y métricas / Decision and metrics
# -----------------------------------------------------------------------------
def decidir(P, t34, t0):
    # 1) P3+P4 >= t34 -> 3/4 ; 2) P0 >= t0 -> 0 ; 3) si no, 1 o 2 / else 1 or 2
    pred = np.where(P[:, 1] >= P[:, 2], 1, 2)
    pred = np.where(P[:, 0] >= t0, 0, pred)
    return np.where(P[:, 3] + P[:, 4] >= t34, np.where(P[:, 3] >= P[:, 4], 3, 4), pred)


def metricas(validos, pred):
    # validos: conjunto de etiquetas aceptadas por fila (dobles) / accepted labels per row
    ok = np.array([p in v for p, v in zip(pred, validos)])
    m = {f"recall_{c}": ok[[c in v for v in validos]].mean() if any(c in v for v in validos) else np.nan
         for c in CLASES}
    es34 = np.array([bool(v & RECLUT) for v in validos])
    pr34 = np.isin(pred, list(RECLUT))
    m["recall_34"] = pr34[es34].mean() if es34.any() else np.nan
    m["precision_34"] = es34[pr34].mean() if pr34.any() else np.nan
    m["precision_0"] = np.array([0 in v for v in validos])[pred == 0].mean() if (pred == 0).any() else np.nan
    m["exactitud"] = ok.mean()
    return m


def puntaje(m):
    a, b = m["recall_0"], m["recall_34"]
    if OBJETIVO == "minimo":
        return min(a, b)
    if OBJETIVO == "armonica":
        return 0.0 if a + b == 0 else 2 * a * b / (a + b)
    return (a + b) / 2


def buscar_umbrales(P, validos):
    # Rejilla de umbrales respetando pisos / Threshold grid honoring floors
    mejor, libre = None, None
    for t34 in REJILLA:
        for t0 in REJILLA:
            m = metricas(validos, decidir(P, t34, t0))
            cand = (puntaje(m), t34, t0, m)
            libre = cand if libre is None or cand[0] > libre[0] else libre
            pisos = all(np.isnan(m[k]) or m[k] >= p for k, p in
                        [("recall_1", PISO_RECALL_1), ("recall_2", PISO_RECALL_2)])
            if pisos and (mejor is None or cand[0] > mejor[0]):
                mejor = cand
    if mejor is None:
        print("  ! Ningún umbral cumple los pisos; uso el mejor sin pisos / no threshold meets floors")
        mejor = libre
    return mejor


def texto_metricas(m):
    claves = ["recall_0", "recall_34", "precision_34", "precision_0",
              "recall_1", "recall_2", "recall_3", "recall_4", "exactitud"]
    return "\n".join(f"  {k:<13} {m[k]:.3f}" for k in claves)


# -----------------------------------------------------------------------------
# 4. Flujo principal / Main flow
# -----------------------------------------------------------------------------
def separar(y, grupos):
    # Train/test ~70/30 estratificado (y agrupado por autor) / stratified (and grouped by author)
    k = 10
    if AGRUPAR_POR_AUTOR:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEMILLA)
        folds = [te for _, te in cv.split(np.zeros(len(y)), y, grupos)]
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEMILLA)
        folds = [te for _, te in cv.split(np.zeros(len(y)), y)]
    test = np.sort(np.concatenate(folds[:round(k * PROP_TEST)]))
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, test


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    reporte = []

    def log(s=""):
        print(s)
        reporte.append(s)

    # Datos / Data
    df = pd.read_parquet(RUTA_PARQUET).reset_index(drop=True)
    df["video_id"] = df["video_id"].astype(str)
    log(f"Posts: {len(df)}  |  grupos: {df['grupo'].value_counts().sort_index().to_dict()}")
    validos = [frozenset({g} | ({int(s)} if pd.notna(s) else set()))
               for g, s in zip(df["grupo"], df["grupo_segundo"])]
    y = df["grupo"].astype(int).values

    # Imágenes / Images
    img = variables_imagen(df["video_id"].tolist(), N_IMAGENES)
    X = construir_tabla(df, img)

    # Split 70/30
    grupos = df["autor"].where(df["autor"] != "", df["video_id"]).values
    tr, te = separar(y, grupos)
    log(f"Train: {len(tr)}  Test: {len(te)}  (semilla / seed {SEMILLA}; "
        f"autores en común / shared authors: {len(set(grupos[tr]) & set(grupos[te]))})")
    log(f"  grupos train: {pd.Series(y[tr]).value_counts().sort_index().to_dict()}")
    log(f"  grupos test:  {pd.Series(y[te]).value_counts().sort_index().to_dict()}")

    # CV en train -> probabilidades fuera de muestra / CV in train -> out-of-fold probabilities
    Xtr, ytr, gtr = X.iloc[tr].reset_index(drop=True), y[tr], grupos[tr]
    vtr = [validos[i] for i in tr]
    cv = (StratifiedGroupKFold(N_FOLDS_CV, shuffle=True, random_state=SEMILLA) if AGRUPAR_POR_AUTOR
          else StratifiedKFold(N_FOLDS_CV, shuffle=True, random_state=SEMILLA))
    P = np.zeros((len(tr), len(CLASES)))
    for k, (a, b) in enumerate(cv.split(Xtr, ytr, gtr), 1):
        P[b] = probabilidades(crear_modelo(Xtr).fit(Xtr.iloc[a], ytr[a]), Xtr.iloc[b])
        print(f"  fold {k}/{N_FOLDS_CV}")

    # Umbrales / Thresholds
    s, t34, t0, m_cv = buscar_umbrales(P, vtr)
    log("\n=== CV en train / CV on train ===")
    log("Sin umbrales (argmax):")
    log(texto_metricas(metricas(vtr, np.array(CLASES)[P.argmax(1)])))
    log(f"Con umbrales / With thresholds  t34={t34}  t0={t0}  ({OBJETIVO}={s:.3f}):")
    log(texto_metricas(m_cv))

    # Modelo final en train, evaluación única en test / Final fit on train, single test eval
    modelo = crear_modelo(Xtr).fit(Xtr, ytr)
    Xte = X.iloc[te].reset_index(drop=True)
    Pte = probabilidades(modelo, Xte)
    pred = decidir(Pte, t34, t0)
    vte = [validos[i] for i in te]
    log("\n=== TEST (30%) ===")
    log(texto_metricas(metricas(vte, pred)))
    # Matriz: si acierta una etiqueta doble, cuenta como esa / doubles resolved in model's favor
    real = np.array([p if p in v else min(v) for p, v in zip(pred, vte)])
    cm = pd.DataFrame(confusion_matrix(real, pred, labels=CLASES),
                      index=[f"real_{c}" for c in CLASES], columns=[f"pred_{c}" for c in CLASES])
    log("\nMatriz de confusión / Confusion matrix:")
    log(cm.to_string())

    # Variables con más peso / Top-weighted features
    nombres = modelo.named_steps["pre"].get_feature_names_out()
    coef = modelo.named_steps["clf"].coef_
    log("\n=== Variables con más peso por grupo / Top features per group ===")
    for j, c in enumerate(modelo.classes_):
        orden = [i for i in np.argsort(coef[j])[::-1] if "__img_" not in nombres[i]][:20]
        log(f"Grupo {c}: " + ", ".join(nombres[i].split("__", 1)[1] for i in orden))
    if USAR_SONDAS_IMAGEN:
        log("\nPeso de las sondas de imagen / Image probe weights:")
        idx = [i for i, n in enumerate(nombres) if "__sonda_" in n]
        tabla = pd.DataFrame(coef[:, idx].T, index=[nombres[i].split("sonda_")[1] for i in idx],
                             columns=[f"g{c}" for c in modelo.classes_]).round(2)
        log(tabla.to_string())

    # Guardar / Save
    out = df.iloc[te][["video_id", "grupo", "grupo_segundo"]].reset_index(drop=True)
    out["pred"] = pred
    out["acierto"] = [p in v for p, v in zip(pred, vte)]
    for c in CLASES:
        out[f"p_{c}"] = Pte[:, c].round(4)
    out.to_csv(os.path.join(CARPETA_SALIDA, "predicciones_test.csv"), index=False)
    joblib.dump({"modelo": modelo, "t34": t34, "t0": t0, "n_imagenes": N_IMAGENES},
                os.path.join(CARPETA_SALIDA, "baby_model.joblib"))
    with open(os.path.join(CARPETA_SALIDA, "reporte.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(reporte))
    print(f"\nListo / Done -> {CARPETA_SALIDA}/")


if __name__ == "__main__":
    main()

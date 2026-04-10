"""
generar_informe.py
==================
Genera un informe PDF de contratos SECOP I + II usando búsqueda semántica
y lo envía por correo.

Requiere en la carpeta data/:
  - embeddings.npy      (generado por generar_embeddings.py)
  - df_index.parquet    (generado por generar_embeddings.py)
  - secop1.parquet      (datos originales SECOP I)
  - secop2.parquet      (datos originales SECOP II)

Variables de entorno (GitHub Secrets):
  EMAIL_SENDER    → correo remitente Gmail (ej: tucorreo@gmail.com)
  EMAIL_PASSWORD  → contraseña de aplicación de Google (16 chars sin espacios)
  EMAIL_TO        → destinatario(s) separados por coma
  SMTP_HOST       → smtp.gmail.com  (default)
  SMTP_PORT       → 587             (default)
"""

# ── Dependencias ──────────────────────────────────────────────────────────────
# pip install sentence-transformers scikit-learn pandas numpy matplotlib pyarrow

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURA AQUÍ
# ════════════════════════════════════════════════════════════════════════════

# Palabras clave que definen la búsqueda semántica
PALABRAS_CLAVE = [
    "publicidad", "campaña", "marketing", "mercadeo",
    "radio", "television", "televisión", "redes sociales",
    "instagram", "tiktok", "influenciador", "influencer",
    "creador de contenido", "pauta", "medios", "comunicación",
    "prensa", "digital", "audiovisual", "producción audiovisual",
    "branding", "estrategia digital", "obras",
]

# Umbral de similitud coseno (0.0–1.0)
# Más bajo  → más resultados, menos precisión
# Más alto  → menos resultados, más precisión
# Recomendado: entre 0.28 y 0.40
UMBRAL_SIMILITUD = 0.32

# Rutas de archivos
EMBEDDINGS_PATH = "data/embeddings.npy"
DF_INDEX_PATH   = "data/df_index.parquet"
SECOP1_PATH     = "data/secop1.parquet"
SECOP2_PATH     = "data/secop2.parquet"
PDF_PATH        = "data/informe_secop_javeriana.pdf"

# Columnas de detalle por fuente (para enriquecer resultados del índice)
DETALLE_S1 = {
    "col_valor":   ["cuantia_contrato", "valor", "cuantia", "monto"],
    "col_entidad": ["nombre_entidad", "entidad"],
    "col_objeto":  ["objeto_del_contrato", "descripcion_del_proceso",
                    "objeto_a_contratar", "detalle_del_objeto_a_contratar"],
    "col_fecha":   ["fecha_de_publicacion", "fecha_publicacion", "fecha"],
    "col_link":    ["ruta_proceso_en_secop_i", "url", "link", "ruta"],
    "col_nivel":   ["nivel_entidad", "orden", "nivel"],
    "col_id":      ["iud", "id"],
    "col_estado":  ["estado_del_proceso"],
    "estado_ok":   "convocado",
}

DETALLE_S2 = {
    "col_valor":   ["valor_del_contrato", "valor_total_adjudicacion", "precio_base", "valor"],
    "col_entidad": ["entidad", "nombre_entidad"],
    "col_objeto":  ["detalle_del_objeto_a_contratar", "objeto_del_contrato_a_la",
                    "objeto_a_contratar", "nombre_del_procedimiento",
                    "descripci_n_del_procedimiento", "descripcion"],
    "col_fecha":   ["fecha_de_publicacion", "fecha_publicacion", "fecha"],
    "col_link":    ["urlproceso", "url", "link"],
    "col_nivel":   ["nivel_entidad", "orden", "nivel", "tipo_entidad"],
    "col_id":      ["id_del_proceso", "id"],
    "col_estado":  ["estado_del_procedimiento"],
    "estado_ok":   "publicado",
}

# Colores institucionales
COLOR_BLUE = "#003087"
COLOR_RED  = "#e63946"

# ════════════════════════════════════════════════════════════════════════════


# ── Helpers ───────────────────────────────────────────────────────────────────

def detectar_col(df, candidatos):
    """Detecta la primera columna disponible de una lista de candidatos."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidatos:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
        for cl, c in cols_lower.items():
            if cand.lower() in cl:
                return c
    return None

def fmt_peso(val):
    try:
        return f"$ {float(val):,.0f}"
    except Exception:
        return "—"

def truncar(texto, n):
    s = str(texto) if texto else "—"
    return s[:n] + "…" if len(s) > n else s


# ── PASO 1: Búsqueda semántica ────────────────────────────────────────────────
print("=" * 60)
print("📊 GENERANDO INFORME SECOP I + II - JAVERIANA")
print("=" * 60)

# Verificar archivos necesarios
for ruta in [EMBEDDINGS_PATH, DF_INDEX_PATH]:
    if not os.path.exists(ruta):
        print(f"❌ Archivo no encontrado: {ruta}")
        print("   Asegúrate de haber corrido generar_embeddings.py primero.")
        exit(1)

print("\n🤖 Cargando índice semántico...")
df_index   = pd.read_parquet(DF_INDEX_PATH)
embeddings = np.load(EMBEDDINGS_PATH)
print(f"   Contratos en índice : {len(df_index):,}")
print(f"   Shape embeddings    : {embeddings.shape}")

print("\n🔍 Generando embedding de consulta...")
consulta  = " ".join(PALABRAS_CLAVE)
model     = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
query_emb = model.encode([consulta], normalize_embeddings=True)

similitudes          = cosine_similarity(query_emb, embeddings)[0]
df_index["_similitud"] = similitudes

df_relevantes = (
    df_index[df_index["_similitud"] >= UMBRAL_SIMILITUD]
    .copy()
    .sort_values("_similitud", ascending=False)
    .reset_index(drop=True)
)

print(f"\n✅ Contratos relevantes (umbral={UMBRAL_SIMILITUD}): {len(df_relevantes):,}")
print(f"   SECOP I : {(df_relevantes['_fuente'] == 'SECOP I').sum():,}")
print(f"   SECOP II: {(df_relevantes['_fuente'] == 'SECOP II').sum():,}")

if df_relevantes.empty:
    print("\nℹ️  Sin contratos relevantes. Baja UMBRAL_SIMILITUD e intenta de nuevo.")
    exit(0)


# ── PASO 2: Enriquecer con columnas del parquet original ──────────────────────
# df_index puede no tener todas las columnas (valor, fecha, link, nivel…)
# Las buscamos en los parquets originales y hacemos merge por id_del_proceso / iud

print("\n📦 Enriqueciendo con datos originales...")

dfs_enriquecidos = []

for fuente, cfg, path in [
    ("SECOP I",  DETALLE_S1, SECOP1_PATH),
    ("SECOP II", DETALLE_S2, SECOP2_PATH),
]:
    subset = df_relevantes[df_relevantes["_fuente"] == fuente].copy()
    if subset.empty:
        continue

    if not os.path.exists(path):
        print(f"   ⚠️  No se encontró {path}. Se usará solo el índice para {fuente}.")
        # Agregar columnas dummy para que el resto del código no falle
        for alias in ["_valor", "_entidad", "_objeto", "_fecha", "_link", "_nivel", "_id"]:
            subset[alias] = None
        dfs_enriquecidos.append(subset)
        continue

    df_orig = pd.read_parquet(path)

    # Filtrar por estado (solo convocados/publicados)
    col_est = detectar_col(df_orig, cfg["col_estado"])
    if col_est:
        df_orig = df_orig[
            df_orig[col_est].fillna("").str.lower().str.strip() == cfg["estado_ok"]
        ].copy()

    # Detectar columnas útiles
    col_id      = detectar_col(df_orig, cfg["col_id"])
    col_valor   = detectar_col(df_orig, cfg["col_valor"])
    col_entidad = detectar_col(df_orig, cfg["col_entidad"])
    col_objeto  = detectar_col(df_orig, cfg["col_objeto"])
    col_fecha   = detectar_col(df_orig, cfg["col_fecha"])
    col_link    = detectar_col(df_orig, cfg["col_link"])
    col_nivel   = detectar_col(df_orig, cfg["col_nivel"])

    print(f"   {fuente} — id={col_id} valor={col_valor} entidad={col_entidad} fecha={col_fecha}")

    # Merge: usar id_del_proceso del índice (columna común guardada en generar_embeddings)
    col_id_index = "id_del_proceso" if "id_del_proceso" in subset.columns else None

    if col_id_index and col_id and col_id_index in subset.columns and col_id in df_orig.columns:
        cols_a_traer = {c: c for c in [col_valor, col_entidad, col_objeto,
                                        col_fecha, col_link, col_nivel] if c}
        df_merge = df_orig[[col_id] + list(cols_a_traer.keys())].copy()
        df_merge = df_merge.rename(columns={col_id: col_id_index})
        subset   = subset.merge(df_merge, on=col_id_index, how="left", suffixes=("", "_orig"))

    # Crear columnas alias normalizadas
    subset["_valor"]   = pd.to_numeric(subset[col_valor],   errors="coerce") if col_valor   and col_valor   in subset.columns else np.nan
    subset["_entidad"] = subset[col_entidad].fillna("—")  if col_entidad and col_entidad in subset.columns else "—"
    subset["_objeto"]  = subset[col_objeto].fillna("—")   if col_objeto  and col_objeto  in subset.columns else subset.get("_texto", "—")
    subset["_fecha"]   = pd.to_datetime(subset[col_fecha], errors="coerce") if col_fecha   and col_fecha   in subset.columns else pd.NaT
    subset["_link"]    = subset[col_link].fillna("—")     if col_link    and col_link    in subset.columns else "—"
    subset["_nivel"]   = subset[col_nivel].fillna("—")    if col_nivel   and col_nivel   in subset.columns else "—"
    subset["_id"]      = subset[col_id_index].astype(str) if col_id_index and col_id_index in subset.columns else "—"

    dfs_enriquecidos.append(subset)

df_total = pd.concat(dfs_enriquecidos, ignore_index=True)
fuente_counts = df_total["_fuente"].value_counts()

print(f"\n📦 Total para informe: {len(df_total):,} contratos")
for f, n in fuente_counts.items():
    print(f"   • {f}: {n:,}")


# ── PASO 3: KPIs ──────────────────────────────────────────────────────────────
monto_total     = df_total["_valor"].sum()
monto_prom      = df_total["_valor"].mean()
monto_max       = df_total["_valor"].max()
total_contratos = len(df_total)
n_ultimo_mes    = int((df_total["_fecha"] >= pd.Timestamp(date.today() - timedelta(days=30))).sum())

top_entidades = (
    df_total.dropna(subset=["_valor"])
    .groupby("_entidad")["_valor"].sum()
    .nlargest(5)
)

nivel_counts = (
    df_total["_nivel"]
    .replace("—", np.nan)
    .dropna()
    .value_counts()
)

df_total["_mes"] = df_total["_fecha"].dt.to_period("M")
hist_meses = (
    df_total.groupby("_mes")["_valor"].count()
    .reset_index()
    .rename(columns={"_valor": "Contratos"})
)
hist_meses["Mes"] = hist_meses["_mes"].astype(str)
hist_meses = hist_meses[hist_meses["Mes"] != "NaT"].tail(18)  # últimos 18 meses

df_top10 = (
    df_total.dropna(subset=["_valor"])
    .nlargest(10, "_valor")
    if not df_total["_valor"].isna().all()
    else df_total.head(10)
)

hoy_str = date.today().strftime("%d/%m/%Y")


# ── PASO 4: Generar PDF ───────────────────────────────────────────────────────
print("\n📄 Generando PDF...")
os.makedirs("data", exist_ok=True)

with PdfPages(PDF_PATH) as pdf:

    # ── Portada ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(COLOR_BLUE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(COLOR_BLUE)
    ax.axis("off")
    ax.text(0.5, 0.74, "INFORME SECOP I + II", color="white", fontsize=36,
            fontweight="bold", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.62, "Oportunidades de Publicidad y Marketing", color="#aac8ff",
            fontsize=18, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.52, "Universidad Javeriana", color="white", fontsize=22,
            fontweight="bold", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.39, f"Generado: {hoy_str}", color="#aac8ff", fontsize=14,
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.31, "SECOP I: Convocado  ·  SECOP II: Publicado",
            color="#aac8ff", fontsize=12, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.22,
            "  ·  ".join([f"{f}: {n:,} contratos" for f, n in fuente_counts.items()]),
            color="white", fontsize=13, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.10,
            f"Búsqueda semántica · umbral similitud: {UMBRAL_SIMILITUD}",
            color="#aac8ff", fontsize=10, ha="center", va="center", transform=ax.transAxes)
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("KPIs Principales · SECOP I + II", fontsize=18,
                 fontweight="bold", color=COLOR_BLUE, y=0.98)
    kpis = [
        ("# Contratos totales",  f"{total_contratos:,}"),
        ("Monto Total",          fmt_peso(monto_total)),
        ("Monto Promedio",       fmt_peso(monto_prom)),
        ("Contrato Más Grande",  fmt_peso(monto_max)),
        ("Contratos (30 días)",  f"{n_ultimo_mes:,}"),
        ("Fuentes",              "\n".join([f"{f}: {n:,}" for f, n in fuente_counts.items()])),
    ]
    for ax, (lbl, val) in zip(axes.flat, kpis):
        ax.set_facecolor("#f0f4ff")
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_BLUE)
            spine.set_linewidth(2)
        ax.axis("off")
        ax.text(0.5, 0.60, val, ha="center", va="center", fontsize=15,
                fontweight="bold", color=COLOR_BLUE, transform=ax.transAxes)
        ax.text(0.5, 0.22, lbl, ha="center", va="center", fontsize=11,
                color="#555", transform=ax.transAxes)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

    # ── SECOP I vs II ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SECOP I vs SECOP II", fontsize=16, fontweight="bold", color=COLOR_BLUE)
    fuentes_names = list(fuente_counts.index)
    fuentes_vals  = list(fuente_counts.values)
    colores_f     = [COLOR_BLUE, COLOR_RED]

    ax1.bar(fuentes_names, fuentes_vals, color=colores_f[:len(fuentes_names)],
            edgecolor="white", width=0.5)
    ax1.set_title("# Contratos por fuente", fontsize=13, color=COLOR_BLUE)
    ax1.set_ylabel("# Contratos")
    for spine in ["top", "right"]: ax1.spines[spine].set_visible(False)
    for i, v in enumerate(fuentes_vals):
        ax1.text(i, v + max(fuentes_vals) * 0.02, f"{v:,}",
                 ha="center", fontsize=11, fontweight="bold")

    montos_fuente = [
        df_total[df_total["_fuente"] == f]["_valor"].sum()
        for f in fuentes_names
    ]
    ax2.bar(fuentes_names, montos_fuente, color=colores_f[:len(fuentes_names)],
            edgecolor="white", width=0.5)
    ax2.set_title("Monto total por fuente ($)", fontsize=13, color=COLOR_BLUE)
    ax2.set_ylabel("Monto ($)")
    for spine in ["top", "right"]: ax2.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Nacional vs Territorial ───────────────────────────────────────────────
    if not nivel_counts.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Distribución por Nivel de Entidad (SECOP I + II)",
                     fontsize=16, fontweight="bold", color=COLOR_BLUE)
        colores_n = [COLOR_BLUE, COLOR_RED, "#4cc9f0", "#7209b7", "#f72585"]
        wedges, texts, autotexts = ax1.pie(
            nivel_counts.values, labels=None, autopct="%1.1f%%", startangle=90,
            colors=colores_n[:len(nivel_counts)],
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(11); at.set_fontweight("bold")
        ax1.set_title("Gráfico de Anillo", fontsize=13, color=COLOR_BLUE)
        handles = [mpatches.Patch(color=colores_n[i], label=f"{k}: {v:,}")
                   for i, (k, v) in enumerate(nivel_counts.items())]
        ax1.legend(handles=handles, loc="lower center", fontsize=10, frameon=False)
        ax2.barh(nivel_counts.index[::-1], nivel_counts.values[::-1],
                 color=colores_n[:len(nivel_counts)][::-1], edgecolor="white")
        ax2.set_title("Cantidad de Contratos por Nivel", fontsize=13, color=COLOR_BLUE)
        ax2.set_xlabel("# Contratos")
        for spine in ["top", "right"]: ax2.spines[spine].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ── Histograma por mes ────────────────────────────────────────────────────
    if not hist_meses.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle("Número de Contratos por Mes (SECOP I + II)",
                     fontsize=16, fontweight="bold", color=COLOR_BLUE)
        bars = ax.bar(hist_meses["Mes"], hist_meses["Contratos"],
                      color=COLOR_BLUE, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fontsize=9, padding=3)
        ax.set_xlabel("Mes de publicación")
        ax.set_ylabel("# Contratos")
        plt.xticks(rotation=45, ha="right")
        for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ── Top 5 entidades ───────────────────────────────────────────────────────
    if not top_entidades.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle("Top 5 Entidades por Monto Total (SECOP I + II)",
                     fontsize=16, fontweight="bold", color=COLOR_BLUE)
        etiquetas = [truncar(e, 50) for e in top_entidades.index]
        colores5  = [COLOR_BLUE, COLOR_RED, "#4cc9f0", "#7209b7", "#f72585"]
        ax.barh(etiquetas[::-1], top_entidades.values[::-1],
                color=colores5[:len(etiquetas)][::-1], edgecolor="white")
        ax.set_xlabel("Monto total ($)")
        for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ── Similitud semántica (distribución) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Distribución de Similitud Semántica de los Contratos Encontrados",
                 fontsize=14, fontweight="bold", color=COLOR_BLUE)
    ax.hist(df_total["_similitud"].dropna(), bins=30,
            color=COLOR_BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(UMBRAL_SIMILITUD, color=COLOR_RED, linestyle="--", linewidth=2,
               label=f"Umbral: {UMBRAL_SIMILITUD}")
    ax.set_xlabel("Similitud coseno"); ax.set_ylabel("# Contratos")
    ax.legend(fontsize=11)
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Tabla top 10 ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("Top 10 Contratos Más Grandes (SECOP I + II)",
                 fontsize=16, fontweight="bold", color=COLOR_BLUE)
    ax.axis("off")
    tabla_data = []
    for _, row in df_top10.iterrows():
        tabla_data.append([
            truncar(row.get("_id",       "—"), 18),
            fmt_peso(row.get("_valor",   None)),
            truncar(row.get("_entidad",  "—"), 35),
            truncar(row.get("_objeto",   row.get("_texto", "—")), 55),
            row.get("_fuente", "—"),
        ])
    tbl = ax.table(
        cellText=tabla_data,
        colLabels=["ID", "Valor ($)", "Entidad", "Objeto (resumen)", "Fuente"],
        cellLoc="left", loc="center",
        colWidths=[0.13, 0.15, 0.28, 0.37, 0.07],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(COLOR_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        cell.set_edgecolor("#dde")
    pdf.savefig(fig)
    plt.close(fig)

    # ── Tabla top 20 por similitud semántica ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.suptitle("Top 20 Contratos Más Relevantes por Similitud Semántica",
                 fontsize=14, fontweight="bold", color=COLOR_BLUE)
    ax.axis("off")
    df_top20_sem = df_total.head(20)
    tabla_sem = []
    for _, row in df_top20_sem.iterrows():
        tabla_sem.append([
            truncar(row.get("_entidad", "—"), 30),
            truncar(row.get("_objeto",  row.get("_texto", "—")), 60),
            f"{row.get('_similitud', 0):.3f}",
            fmt_peso(row.get("_valor", None)),
            row.get("_fuente", "—"),
        ])
    tbl2 = ax.table(
        cellText=tabla_sem,
        colLabels=["Entidad", "Objeto (resumen)", "Similitud", "Valor ($)", "Fuente"],
        cellLoc="left", loc="center",
        colWidths=[0.22, 0.48, 0.09, 0.13, 0.08],
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(7.5)
    tbl2.scale(1, 1.4)
    for (r, c), cell in tbl2.get_celld().items():
        if r == 0:
            cell.set_facecolor(COLOR_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        cell.set_edgecolor("#dde")
    pdf.savefig(fig)
    plt.close(fig)

    # ── Pie de página / palabras clave ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    ax.text(0.5, 0.80, "Palabras clave utilizadas en la búsqueda semántica:",
            ha="center", fontsize=12, fontweight="bold", color=COLOR_BLUE,
            transform=ax.transAxes)
    ax.text(0.5, 0.52, ", ".join(PALABRAS_CLAVE), ha="center", fontsize=10,
            color="#444", transform=ax.transAxes, wrap=True)
    ax.text(0.5, 0.22,
            f"Modelo: paraphrase-multilingual-MiniLM-L12-v2  ·  "
            f"Umbral similitud: {UMBRAL_SIMILITUD}  ·  "
            f"Datos: datos.gov.co  ·  {hoy_str}",
            ha="center", fontsize=9, color="#999", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)

print(f"✅ PDF generado: {PDF_PATH}")


# ── PASO 5: Enviar correo ─────────────────────────────────────────────────────
print("\n📧 ENVIANDO CORREO...")

EMAIL_SENDER   = os.environ.get("EMAIL_SENDER",   "almacenbiancasilva11@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "yylxvoxojnjlvfae")
EMAIL_TO_RAW   = os.environ.get("EMAIL_TO",       "juansilva11@outlook.com, danimorav05@gmail.com")
EMAIL_TO       = [e.strip() for e in EMAIL_TO_RAW.split(",") if e.strip()]
SMTP_HOST      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.environ.get("SMTP_PORT", 587))

if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_TO:
    print("⚠️  Variables de correo no configuradas como variables de entorno.")
    print(f"   PDF disponible en: {PDF_PATH}")
    print("   Configura EMAIL_SENDER, EMAIL_PASSWORD y EMAIL_TO como secrets en GitHub.")
    exit(0)

hoy_legible     = date.today().strftime("%d de %B de %Y")
fuentes_resumen = "  |  ".join([f"{f}: {n:,}" for f, n in fuente_counts.items()])

msg            = MIMEMultipart()
msg["Subject"] = f"📊 Informe SECOP Javeriana – {hoy_legible} – {total_contratos:,} contratos"
msg["From"]    = EMAIL_SENDER
msg["To"]      = ", ".join(EMAIL_TO)

cuerpo = (
    f"Hola,\n\n"
    f"Adjunto el informe de contratos SECOP relevantes para la "
    f"Universidad Javeriana (publicidad, marketing, medios y obras).\n\n"
    f"📌 Resumen:\n"
    f"   • Contratos encontrados : {total_contratos:,}\n"
    f"   • {fuentes_resumen}\n"
    f"   • Monto total           : {fmt_peso(monto_total)}\n"
    f"   • Monto promedio        : {fmt_peso(monto_prom)}\n"
    f"   • Contratos (30 días)   : {n_ultimo_mes:,}\n\n"
    f"Búsqueda semántica con umbral de similitud: {UMBRAL_SIMILITUD}\n"
    f"Modelo: paraphrase-multilingual-MiniLM-L12-v2\n"
    f"Fuente: datos.gov.co · SECOP I (Convocado) + SECOP II (Publicado)\n\n"
    f"Generado automáticamente el {hoy_legible}.\n"
)
msg.attach(MIMEText(cuerpo, "plain"))

with open(PDF_PATH, "rb") as f:
    part = MIMEBase("application", "octet-stream")
    part.set_payload(f.read())
encoders.encode_base64(part)
part.add_header(
    "Content-Disposition",
    f"attachment; filename=informe_secop_{date.today()}.pdf"
)
msg.attach(part)

try:
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_TO, msg.as_string())
    print(f"✅ Correo enviado a: {', '.join(EMAIL_TO)}")
except Exception as e:
    print(f"❌ Error al enviar correo: {e}")
    print(f"   PDF guardado igualmente en: {PDF_PATH}")
    exit(1)

print("\n✅ Proceso finalizado correctamente.")
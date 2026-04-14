import pandas as pd
import numpy as np
import geopandas as gpd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tabulate as tabulate
import unicodedata
from shapely import wkb
import matplotlib.pyplot as plt
import math
import os
from datetime import date, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import requests
from io import BytesIO
from PIL import Image as PILImage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
RUTA_GRAFICOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(RUTA_GRAFICOS, exist_ok=True)
DIC_COLORES = {
    'verde':    ["#009966"],
    'ro_am_na': ["#FFE9C5", "#F7B261", "#D8841C", "#dd722a", "#C24C31", "#BC3B26"],
    'az_verd':  ["#CBECEF", "#81D3CD", "#0FB7B3", "#009999"],
    'ax_viol':  ["#D9D9ED", "#2F399B", "#1A1F63", "#262947"],
    'ofiscal':  ["#F9F9F9", "#2635bf"],
    'colors':   ["#262947", "#009999", "#81D3CD", "#dd722a", "#F7B261"]
}
 
# Color principal de cada fuente  (coherente en TODOS los gráficos)
COLOR_SECOP1 = "#2F399B"   # azul-violeta  — ax_viol[1]
COLOR_SECOP2 = "#D8841C"   # naranja medio — ro_am_na[2]
 
# Colores para el donut de tipo de entidad
COLOR_ORDEN = {
    'Territorial':          "#2F399B",   # teal  — az_verd[3]
    'Nacional':             "#262947",   # azul muy oscuro — ax_viol[3]
    'Corporación Autónoma': "#dd722a",   # naranja cálido  — ro_am_na[3]
    'No Definido':          "#D9D9ED",   # gris lavanda    — ax_viol[0]
}
 
# Colores institucionales del PDF
COLOR_BLUE   = "#2635bf"   # ofiscal[1]
COLOR_ACCENT = "#D8841C"   # ro_am_na[2]  línea separadora
 
# ============================================================
# 🔧 CAMBIO 3 — LOGO: descarga única y función auxiliar
# ============================================================
 
def cargar_logo(ruta: str, max_height_px: int = 60):
    try:
        img = PILImage.open(ruta).convert("RGBA")
        ratio = max_height_px / img.height
        new_size = (int(img.width * ratio), max_height_px)
        img = img.resize(new_size, PILImage.LANCZOS)
        arr = np.array(img)
        print(f"✅ Logo cargado correctamente desde: {ruta}")  # ← ANTES del return
        return arr
    except Exception as e:
        print(f"⚠️ No se pudo cargar el logo: {e}")
        return None
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_logo = os.path.join(BASE_DIR, "data", "logo_sapo.png")
print(f"Buscando logo en: {ruta_logo}")
print(f"¿Existe el archivo? {os.path.exists(ruta_logo)}")
LOGO_ARRAY = cargar_logo(ruta_logo)
print(f"LOGO_ARRAY es None: {LOGO_ARRAY is None}")
def poner_logo(fig, logo_array=None, x=0.01, y=0.005, zoom=0.3):
    """Pega el logo en la esquina inferior-izquierda de la figura."""
    if logo_array is None:
        logo_array = LOGO_ARRAY  # ← lee la variable global en tiempo de ejecución
    if logo_array is None:
        return
    imagebox = OffsetImage(logo_array, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        (x, y),
        xycoords="figure fraction",
        frameon=False,
        box_alignment=(0, 0),
    )
    fig.add_artist(ab)
def guardar_grafico(nombre):
    ruta = os.path.join(RUTA_GRAFICOS, f"{nombre}.png")
    plt.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"✅ Guardado: {ruta}")
muns = pd.read_parquet("https://raw.githubusercontent.com/PinchCapybara21/api_fiscal/main/muns.parquet")
model      = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings2 = np.load("data/embeddings.npy")
embeddings1 = np.load("data/embeddings1.npy")
df2        = pd.read_parquet("https://raw.githubusercontent.com/Danii114/SECOP_I-II/main/data/secop2.parquet")
df1        = pd.read_parquet("https://raw.githubusercontent.com/Danii114/SECOP_I-II/main/data/secop_2026-04.parquet")
palabras_clave = [
    "influenciador", "prensa", "medios", "redes sociales", "publicidad", "campaña", "marketing", "mercadeo",
    "radio", "television", "televisión",
    "instagram", "tiktok", "influencer",
    "creador de contenido", "pauta", "comunicación", "digital", "audiovisual", "producción audiovisual",
    "branding", "estrategia digital"
]
COLUMNAS_s1 = [
    "nombre_entidad",
    "nit_de_la_entidad",
    "orden_entidad",
    "modalidad_de_contratacion",
    "estado_del_proceso",
    "detalle_del_objeto_a_contratar",
    "fecha_de_cargue_en_el_secop",
    "cuantia_proceso",
    "municipio_entidad",
    "ruta_proceso_en_secop_i",
    "score",
    'departamento_entidad'
    ]
COLUMNAS_s2 = [
    "nit_entidad", "ordenentidad", "entidad",
    "modalidad_de_contratacion", 'estado_del_procedimiento','ciudad_de_la_unidad_de', 'departamento_entidad',
    "descripci_n_del_procedimiento", "fecha_de_publicacion_del",
    "precio_base",
    "urlproceso", "score"]
#aca es con proceso de contratacion, puede que se cambie a CONTRATOS ELECTRONICOS, si es asi, toca cambar el nombre de las columnas a las nuevas y volver a cambairlas a las que
#tenia en el codigo para que siga sirviendo todo el codigo
def buscar_secop2 (query: str, top_k: int = 15) -> pd.DataFrame:
    query_limpio = query.lower().strip() 
    query_emb    = model.encode([query_limpio])
    similitudes  = cosine_similarity(query_emb, embeddings2)[0]
    top_idx      = similitudes.argsort()[-top_k:][::-1]
    cols_presentes = [c for c in COLUMNAS_s2 if c in df2.columns]
    resultado    = df2.iloc[top_idx][cols_presentes].copy()
    resultado["score"] = similitudes[top_idx]
    return resultado
resultados_secop2 = [buscar_secop2(p) for p in palabras_clave]   #aca puse agua de pruebaa, la idea es poner palabras clave
resultados_secop2 = pd.concat(resultados_secop2, ignore_index=True)
resultados_secop2 = resultados_secop2.drop_duplicates(subset=["urlproceso"])
resultados_secop2['fuente'] = 'secop II'
def buscar_secop1 (query: str, top_k: int = 15) -> pd.DataFrame:
    query_limpio = query.lower().strip() 
    query_emb    = model.encode([query_limpio])
    similitudes  = cosine_similarity(query_emb, embeddings1)[0]
    top_idx      = similitudes.argsort()[-top_k:][::-1]
    cols_presentes = [c for c in COLUMNAS_s1 if c in df1.columns]
    resultado    = df1.iloc[top_idx][cols_presentes].copy()
    resultado["score"] = similitudes[top_idx]
    return resultado
resultados_secop1 = [buscar_secop1(p) for p in palabras_clave] #aca puse agua de pruebaa, la idea es poner palabras clave
resultados_secop1 = pd.concat(resultados_secop1, ignore_index=True)
resultados_secop1 = resultados_secop1.drop_duplicates(subset=["ruta_proceso_en_secop_i"])
resultados_secop1['fuente'] = 'secop I'
# print(resultados_secop1)
# print(resultados_secop2)
resultados_secop1 = resultados_secop1.rename(columns={
    'nit_de_la_entidad': 'nit_entidad',  #1
    'nombre_entidad': 'entidad',             #2
    'detalle_del_objeto_a_contratar': 'descripci_n_del_procedimiento', #3
    'orden_entidad': 'ordenentidad', #4
    'modalidad_de_contratacion': 'modalidad_de_contratacion',  #5
    'estado_del_proceso': 'fase',#6
    'fecha_de_cargue_en_el_secop': 'fecha_de_publicacion_del', #7
    'cuantia_proceso': 'precio_base', #8
    'municipio_entidad': 'ciudad_de_la_unidad_de', #9
    'ruta_proceso_en_secop_i': 'urlproceso', #10
    'score': 'score', #11
    'departamento_entidad': 'departamento_entidad', #12
    'fuente': 'fuente' #13
    })
resultados_secop2 = resultados_secop2.rename(columns={
    'estado_del_procedimiento': 'fase', #1
})

todo = pd.concat([resultados_secop1, resultados_secop2], ignore_index=True)
# print(todo.to_string(max_colwidth=None, index=False))

# ─────────────────────────────────────────────
# FILTRO: última semana + estado activo por fuente
# ─────────────────────────────────────────────
todo["fecha_de_publicacion_del"] = pd.to_datetime(
    todo["fecha_de_publicacion_del"], errors="coerce"
)

fecha_corte = pd.Timestamp.today() - pd.Timedelta(weeks=3)

mask_fecha = todo["fecha_de_publicacion_del"] >= fecha_corte

mask_estado = (
    ((todo["fuente"] == "secop I")  & (todo["fase"].str.upper().str.strip() == "CONVOCADO")) |
    ((todo["fuente"] == "secop II") & (todo["fase"].str.upper().str.strip() == "PUBLICADO"))
)

todo = todo[mask_fecha & mask_estado].reset_index(drop=True)

print(f"✅ Registros tras filtro: {len(todo)} "
      f"(SECOP I: {(todo['fuente']=='secop I').sum()} | "
      f"SECOP II: {(todo['fuente']=='secop II').sum()})")

# KPIs
todo["precio_base"] = pd.to_numeric(todo["precio_base"], errors="coerce")
total_contratos    = len(todo)
valor_total        = todo["precio_base"].sum()
valor_promedio     = todo["precio_base"].mean()
valor_mediana      = todo["precio_base"].median()
entidades_unicas   = todo["entidad"].nunique()

part = todo.groupby("fuente").agg(
    n_contratos=("entidad", "count"),
    valor=("precio_base", "sum")
).assign(
    pct_contratos=lambda x: (x["n_contratos"] / total_contratos * 100).round(1),
    pct_valor=lambda x: (x["valor"] / valor_total * 100).round(1)
)

print(f"Total contratos    : {total_contratos}")
print(f"Valor total        : ${valor_total:,.0f}")
print(f"Valor promedio     : ${valor_promedio:,.0f}")
print(f"Mediana            : ${valor_mediana:,.0f}")
print(f"Entidades únicas   : {entidades_unicas}")
print(part)
print(todo["ordenentidad"].unique())
#TORTA
#------------------------------------------
def clasificar_orden(valor):
    if pd.isna(valor):
        return "No Definido"
    v = str(valor).strip()
    if v in ["Nacional", "NACIONAL CENTRALIZADO"]:
        return "Nacional"
    if v in ["TERRITORIAL DEPARTAMENTAL CENTRALIZADO", "TERRITORIAL DEPARTAMENTAL DESCENTRALIZADO",
             "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 2", "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 4",
             "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 5", "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 6",
             "Territorial"]:
        return "Territorial"
    if v == "Corporación Autónoma":
        return "Corporación Autónoma"
    return "No Definido"

todo["orden_clasificado"] = todo["ordenentidad"].fillna("No Definido").apply(clasificar_orden)

orden_contratos = todo.groupby("orden_clasificado").size().reset_index(name="n_contratos")
orden_valor     = todo.groupby("orden_clasificado")["precio_base"].sum().reset_index(name="valor_total")

print(orden_contratos)
print(orden_valor)

### serie de tiempo contratos por mes, dividio secop 1 y 2 aun no esta dividido, se hace en el grafico
todo["fecha"] = pd.to_datetime(todo["fecha_de_publicacion_del"], errors="coerce")
todo["mes"] = todo["fecha"].dt.to_period("M")

contratos_mes = todo.groupby(["mes", "fuente"]).size().reset_index(name="n_contratos")
contratos_mes["mes"] = contratos_mes["mes"].astype(str)

# print(contratos_mes)
#concentracion % % del valor en el Top 5 entidades % del valor en el Top 10
valor_por_entidad = todo.groupby("entidad")["precio_base"].sum().sort_values(ascending=False).reset_index()
valor_por_entidad["pct_valor"] = (valor_por_entidad["precio_base"] / valor_total * 100).round(1)

top5  = valor_por_entidad.head(5)["pct_valor"].sum().round(1)
top10 = valor_por_entidad.head(10)["pct_valor"].sum().round(1)
print("ACA TIENES QUE VEEEEEEEEEEEEEEEEEEEEEER")
print(todo.groupby(['entidad', 'fuente'])['precio_base'].sum().sort_values(ascending=False).reset_index(name='valor_total').head(20))
print(f"Top 5  entidades concentran: {top5}% del valor total")
print(f"Top 10 entidades concentran: {top10}% del valor total")
print(valor_por_entidad.head(10))
print(todo['modalidad_de_contratacion'].unique())

# modalidad de contratación
def clasificar_modalidad(valor):
    if pd.isna(valor):
        return "No Definido"
    v = str(valor).upper().strip()
    if "DIRECTA" in v:
        return "Contratación Directa"
    elif "LICITACION" in v or "LICITACIÓN" in v: 
        return "Licitación Pública"
    elif "MENOR CUANT" in v or "MINIMA CUANTIA" in v or "MÍNIMA CUANTÍA" in v:
        return "Menor Cuantía"
    elif "CONCURSO" in v or "MERITOS" in v:
        return "Concurso de Méritos"
    elif "SUBASTA" in v:
        return "Subasta"
    elif "CONVENIO" in v or "MAS DE DOS PARTES" in v or "MÁS DE DOS PARTES" in v:
        return "Convenios" 
    elif "ESPECIAL" in v or "REGIMEN ESPECIAL" in v or "RÉGIMEN ESPECIAL" in v:
        return "Régimen Especial"
    else:
        return "Otra"

todo["modalidad_clasificada"] = todo["modalidad_de_contratacion"].apply(clasificar_modalidad)

modalidad_contratos = todo.groupby("modalidad_clasificada").size().reset_index(name="n_contratos")
modalidad_valor     = todo.groupby("modalidad_clasificada")["precio_base"].sum().reset_index(name="valor_total")
print(todo.groupby(['modalidad_clasificada', 'fuente'])['fuente'])
print(modalidad_contratos)
print(modalidad_valor)
print(todo["ciudad_de_la_unidad_de"].unique())
print(todo["departamento_entidad"].unique())
print(todo["departamento_entidad"].value_counts())
print(tabulate.tabulate(todo["ciudad_de_la_unidad_de"].value_counts().reset_index(), headers="keys", tablefmt="pretty"))

# C. Distribución de valores
distribucion = todo["precio_base"].dropna()
q1  = distribucion.quantile(0.25)
q3  = distribucion.quantile(0.75)
iqr = q3 - q1
outliers = distribucion[distribucion > q3 + 1.5 * iqr]

print(f"Contratos pequeños (< Q1): {(distribucion < q1).sum()}")
print(f"Contratos gigantes (outliers): {len(outliers)}")
print(f"Valor mínimo : ${distribucion.min():,.0f}")
print(f"Valor máximo : ${distribucion.max():,.0f}")
print(f"Q1: ${q1:,.0f} | Q3: ${q3:,.0f}")
#D
geo_contratos = todo.groupby("ciudad_de_la_unidad_de").size().reset_index(name="n_contratos").sort_values("n_contratos", ascending=False)
geo_valor     = todo.groupby("ciudad_de_la_unidad_de")["precio_base"].sum().reset_index(name="valor_total").sort_values("valor_total", ascending=False)

print(geo_contratos.head(10))
print(geo_valor.head(10))
#8
print("\n🚨 Contratos outliers:")
print(todo[todo["precio_base"] > q3 + 1.5 * iqr][["entidad", "precio_base", "fuente", "urlproceso"]])

"""
De esos datos salen estos gráficos:
De distribución de valores salen dos gráficos: un boxplot que muestra Q1, Q3, mediana y outliers visualmente, y un histograma que muestra cuántos contratos hay en cada rango de valor.
De geografía salen dos gráficos de barras horizontales: top 10 ciudades por número de contratos y top 10 ciudades por valor total. Son más claros que un mapa y ya tienes los datos listos.
De alertas sale una tabla de contratos outliers con entidad, valor y URL, eso no es gráfico sino tabla en el informe.
"""

"""
#HUMMM SOSPECHOSOS DE ACA PA ABAJO
import unicodedata

def normalizar(texto):
    if pd.isna(texto):
        return "no definido"
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    # casos especiales
    texto = texto.replace("bogota d.c.", "bogota").replace("distrito capital de bogota", "bogota")
    return texto

todo["depto_norm"] = todo["departamento_entidad"].apply(normalizar)
muns["depto_norm"] = muns["depto"].apply(normalizar)

# agrupa todo por departamento
geo_deptos = todo.groupby("depto_norm").agg(
    n_contratos=("entidad", "count"),
    valor_total=("precio_base", "sum")
).reset_index()

print(geo_deptos)
muns["geometry"] = muns["geometry"].apply(lambda x: wkb.loads(x))
gdf = gpd.GeoDataFrame(muns, geometry="geometry")

geo_mapa = gdf[["depto_norm", "geometry"]].drop_duplicates(subset="depto_norm")
geo_mapa = geo_mapa.merge(geo_deptos, on="depto_norm", how="left")

colombia = gpd.read_file("https://raw.githubusercontent.com/hvaldivieso/colombia-geojson/master/departamentos.geojson")
print(colombia.columns.tolist())
print(colombia["NOMBRE_DPT"].unique())
"""
#EMPEZAMOS CON GRAFICOS
#tort a 1 -------------------------------------------------------------------------------------------------------
def grafico_torta_orden(todo, fuente, nombre_archivo):
    if todo.empty:
        conteo = pd.Series({"Sin datos": 1})
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Sin contratos\nen este periodo",
                ha="center", va="center", fontsize=14, color="#999",
                transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"Distribución de contratos por tipo de entidad\n{fuente}",
                     fontsize=14, weight='semibold')
        guardar_grafico(nombre_archivo)
        return

    conteo = todo["orden_clasificado"].value_counts().sort_values(ascending=False)

    
    fig, ax = plt.subplots(figsize=(10, 6))

    wedges, _ = ax.pie(
        conteo,
        startangle=90,
        wedgeprops=dict(width=0.38),
        colors=[COLOR_ORDEN.get(cat) for cat in conteo.index]

    )
    total = conteo.sum()

    UMBRAL_PCT = 3.0

    for wedge, val in zip(wedges, conteo):
        ang = (wedge.theta2 + wedge.theta1) / 2
        ang_rad = ang * (math.pi / 180)
        cos_a = math.cos(ang_rad)
        sin_a = math.sin(ang_rad)

        pct = val / total * 100
        label = f"{pct:.1f}%\n({val})"

        if pct < UMBRAL_PCT:
            r_inner = 1.05
            r_text  = 1.22
            x_line, y_line = r_inner * cos_a, r_inner * sin_a
            x_text, y_text = r_text  * cos_a, r_text  * sin_a
            ha = "left" if cos_a >= 0 else "right"

            ax.annotate(
                label,
                xy=(x_line, y_line),
                xytext=(x_text, y_text),
                ha=ha, va="center",
                fontsize=9, weight="bold", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
            )
        else:
            r_text = 0.78
            x_text = r_text * cos_a
            y_text = r_text * sin_a

            ax.text(
                x_text, y_text, label,
                ha="center", va="center",
                fontsize=10, weight="bold", color="#333333"
            )

    centre_circle = plt.Circle((0, 0), 0.58, fc='white')
    fig.gca().add_artist(centre_circle)

    ax.text(0, 0, f"{total:,}\nContratos",
            ha='center', va='center', fontsize=13,
            weight='semibold', color="#222222")

    ax.legend(
        wedges, conteo.index,
        title="Tipo de entidad",
        loc="center left", bbox_to_anchor=(1, 0.5),
        frameon=False, fontsize=10, title_fontsize=11
    )

    ax.set_title(f"Distribución de contratos por tipo de entidad\n{fuente}",
                 fontsize=14, weight='semibold', pad=15, loc='center')

    plt.subplots_adjust(right=0.68)
    guardar_grafico(nombre_archivo)

secop1 = todo[todo["fuente"] == "secop I"]
secop2 = todo[todo["fuente"] == "secop II"]

grafico_torta_orden(secop1, "SECOP I",  "torta_orden_secop1")
grafico_torta_orden(secop2, "SECOP II", "torta_orden_secop2", )
#tora 2 _---------------------------------------------------------------------------------------------------
#torta 2 -------------------------------------------------------------------------------------------------------
def grafico_torta_valor(todo, fuente, nombre_archivo, color="#4C72B0"):
    if todo.empty or todo["precio_base"].sum() == 0:
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.text(0.5, 0.5, "Sin valor contratado\nen este periodo",
                ha="center", va="center", fontsize=14, color="#999",
                transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"Distribución de precio base por tipo de entidad\n{fuente}",
                     fontsize=14, weight='semibold')
        guardar_grafico(nombre_archivo)
        return  
    orden_valor = todo.groupby("orden_clasificado")["precio_base"].sum().reset_index(name="valor_total")
    orden_valor = orden_valor.sort_values(by="valor_total", ascending=False).reset_index(drop=True)

    total_precio = orden_valor["valor_total"].sum()
    UMBRAL_PCT = 3.0
    grandes = orden_valor[orden_valor["valor_total"] / total_precio * 100 >= UMBRAL_PCT].reset_index(drop=True)
    pequeños = orden_valor[orden_valor["valor_total"] / total_precio * 100 < UMBRAL_PCT].reset_index(drop=True)

    filas = []
    pi = 0
    for i, g in grandes.iterrows():
        filas.append(g)
        if pi < len(pequeños):
            filas.append(pequeños.iloc[pi])
            pi += 1

    while pi < len(pequeños):
        filas.append(pequeños.iloc[pi])
        pi += 1

    orden_reorg = pd.DataFrame(filas).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    wedges, _ = ax.pie(
        orden_reorg["valor_total"],
        startangle=90,
        wedgeprops=dict(width=0.38),
        colors=[color] * len(orden_reorg)
    )

    def formato_precio(val):
        val = float(val)
        if val >= 1_000_000_000_000:
            return f"${val/1_000_000_000_000:.1f}B"
        elif val >= 1_000_000_000:
            return f"${val/1_000_000_000:.1f}MM"
        elif val >= 1_000_000:
            return f"${val/1_000_000:.1f}M"
        else:
            return f"${val:,.0f}"

    for wedge, val in zip(wedges, orden_reorg["valor_total"]):
        pct = float(val) / total_precio * 100
        ang = (wedge.theta2 + wedge.theta1) / 2
        ang_rad = ang * math.pi / 180
        cos_a = math.cos(ang_rad)
        sin_a = math.sin(ang_rad)
        label = f"{pct:.1f}%\n{formato_precio(val)}"

        if pct < UMBRAL_PCT:
            r_inner, r_text = 1., 1.1
            ax.annotate(
                label,
                xy=(r_inner * cos_a, r_inner * sin_a),
                xytext=(r_text * cos_a, r_text * sin_a),
                ha="left" if cos_a >= 0 else "right",
                va="center", fontsize=8, weight="bold", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
            )
        else:
            r_text = 0.78
            ax.text(
                r_text * cos_a, r_text * sin_a, label,
                ha="center", va="center",
                fontsize=10, weight="bold", color="#333333"
            )

    centre_circle = plt.Circle((0, 0), 0.58, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.text(0, 0, f"{formato_precio(total_precio)}\nTotal",
            ha="center", va="center", fontsize=13, weight="semibold", color="#222222")
    ax.legend(
        wedges, orden_reorg["orden_clasificado"],
        title="Tipo de entidad",
        loc="center left", bbox_to_anchor=(1, 0.5),
        frameon=False, fontsize=10, title_fontsize=11
    )
    ax.set_title(f"Distribución de precio base por tipo de entidad\n{fuente}",
             fontsize=14, weight="semibold", pad=15)

    plt.subplots_adjust(right=0.65)
    plt.tight_layout()
    guardar_grafico(nombre_archivo)

grafico_torta_valor(secop1, "SECOP I",  "torta_valor_secop1", COLOR_SECOP1)
grafico_torta_valor(secop2, "SECOP II", "torta_valor_secop2", "#D8841C")
#GRAFICO SERIE DE TIEMPO CONTRATO POR QUINCENA  -------------------------------------------------------------------------
def quincena(fecha):
    if pd.isna(fecha):
        return None
    if fecha.day <= 15:
        return fecha.replace(day=1)
    else:
        return fecha.replace(day=16)

todo["quincena"] = todo["fecha"].apply(quincena)

contratos_quincena = todo.groupby(["quincena", "fuente"]).size().reset_index(name="n_contratos")
contratos_quincena = contratos_quincena.dropna(subset=["quincena"]).sort_values("quincena")

contratos_quincena["quincena_str"] = (
    contratos_quincena["quincena"].dt.strftime("%Y-%m") +
    np.where(contratos_quincena["quincena"].dt.day == 1, " Q1", " Q2")
)

# TOP 5 Y 10 -------------------------------------------------------------------------------------------------------
top5_entidades  = valor_por_entidad.head(5)["entidad"]
top10_entidades = valor_por_entidad.head(10)["entidad"]

# Concentración por fuente SECOP
def secop_concentracion(entidades):
    df = todo[todo["entidad"].isin(entidades)]
    return df.groupby("fuente")["precio_base"].sum()

top5_secop_pct  = (secop_concentracion(top5_entidades)  / valor_total * 100).round(1)
top10_secop_pct = (secop_concentracion(top10_entidades) / valor_total * 100).round(1)
s1_vacio = todo[todo["fuente"] == "secop I"].empty
s2_vacio = todo[todo["fuente"] == "secop II"].empty
# Grafico
labels    = ["Top 5", "Top 10"]
plt.figure(figsize=(8, 6))
if s1_vacio or valor_total == 0:
    plt.text(0.5, 0.5, "Sin datos\nen este periodo", ha="center", va="center",
             fontsize=14, color="#999", transform=plt.gca().transAxes)
    plt.axis("off")
else:
    bars1 = plt.bar(labels, 
                [top5_secop_pct.get("secop I", 0), top10_secop_pct.get("secop I", 0)],
                color=COLOR_SECOP1, alpha=0.9) 
    plt.ylabel("% del valor total contratado")
    plt.ylim(0, 100)
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                 f"{height:.1f}%", ha='center', fontsize=11, fontweight='bold')
    plt.grid(axis="y", alpha=0.3, linestyle='--')
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
plt.title("Concentración del valor contratado - SECOP I", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
guardar_grafico('top_5_10_secop_I')


# BARRAS SECOP II
plt.figure(figsize=(8, 6))
if s2_vacio or valor_total == 0:
    plt.text(0.5, 0.5, "Sin datos\nen este periodo", ha="center", va="center",
             fontsize=14, color="#999", transform=plt.gca().transAxes)
    plt.axis("off")
else:
    bars2 = plt.bar(labels,
                    [top5_secop_pct.get("secop II", 0), top10_secop_pct.get("secop II", 0)],
                    color=COLOR_SECOP2, alpha=0.9)
    plt.ylabel("% del valor total contratado")
    plt.ylim(0, 100)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                 f"{height:.1f}%", ha='center', fontsize=11, fontweight='bold')
    plt.grid(axis="y", alpha=0.3, linestyle='--')
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)

plt.title("Concentración del valor contratado - SECOP II", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
guardar_grafico('top_5_10_secop_II')
#GRAFICO DE BARRAS HORIZONTALES DESCOMPOSICION POR ENTIDAD TOP 10 -------------------------------------------------------------------------------------------------------
top_n = 10

valor_por_entidad = (
    todo.groupby("entidad")["precio_base"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
# =============================================
# GRÁFICO 1: TOP 10 ENTIDADES - SOLO SECOP I
# =============================================

# Filtrar solo SECOP I y tomar top 10
top10_secop1 = (
    todo[todo["fuente"] == "secop I"]
    .groupby("entidad")["precio_base"]
    .sum()
    .sort_values(ascending=False)
    .head(top_n)
    .reset_index()
)

top10_secop1["pct"] = top10_secop1["precio_base"] / todo["precio_base"].sum() * 100

fig, ax = plt.subplots(figsize=(12, 8))
if s1_vacio or top10_secop1.empty:
    ax.text(0.5, 0.5, "Sin datos\nen este periodo", ha="center", va="center",
            fontsize=14, color="#999", transform=ax.transAxes)
    ax.axis("off")
else:
    ax.barh(top10_secop1["entidad"], top10_secop1["pct"], color="#2F399B", alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("% del valor total del sistema")
    for i, pct in enumerate(top10_secop1["pct"]):
        ax.text(pct + 0.2, i, f"{pct:.1f}%", va='center', fontweight='bold', fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
ax.set_title("Top 10 Entidades por Valor Contratado - SECOP I", fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
guardar_grafico('top10_secop_I')
# =============================================
# GRÁFICO 2: TOP 10 ENTIDADES - SOLO SECOP II
# =============================================

top10_secop2 = (
    todo[todo["fuente"] == "secop II"]
    .groupby("entidad")["precio_base"]
    .sum()
    .sort_values(ascending=False)
    .head(top_n)
    .reset_index()
)

top10_secop2["pct"] = top10_secop2["precio_base"] / todo["precio_base"].sum() * 100

fig, ax = plt.subplots(figsize=(12, 8))
if s2_vacio or top10_secop2.empty:
    ax.text(0.5, 0.5, "Sin datos\nen este periodo", ha="center", va="center",
            fontsize=14, color="#999", transform=ax.transAxes)
    ax.axis("off")
else:
    ax.barh(top10_secop2["entidad"], top10_secop2["pct"], color=COLOR_SECOP2, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("% del valor total del sistema")
    for i, pct in enumerate(top10_secop2["pct"]):
        ax.text(pct + 0.2, i, f"{pct:.1f}%", va='center', fontweight='bold', fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
ax.set_title("Top 10 Entidades por Valor Contratado - SECOP II", fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
guardar_grafico('top10_secop_II')
# ============================================================
# MODALIDAD POR % — separado por fuente (2 subplots)
# ============================================================

modalidad_contratos["pct"] = (modalidad_contratos["n_contratos"] / modalidad_contratos["n_contratos"].sum() * 100).round(1)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Define un color por fuente
colores_fuente_modalidad = {
    "secop I": "#2F399B",   # azul actual
    "secop II": COLOR_SECOP2  # naranja
}

for ax, fuente in zip(axes, ["secop I", "secop II"]):
    df_f = todo[todo["fuente"] == fuente]
    if df_f.empty:
        ax.text(0.5, 0.5, "Sin datos\nen este periodo", ha="center", va="center",
                fontsize=14, color="#999", transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"Modalidades más utilizadas\n{fuente.upper()}", fontsize=13, fontweight="bold")
        continue 
    conteos = df_f.groupby("modalidad_clasificada").size().sort_values()
    total_f = conteos.sum()
    pcts = (conteos / total_f * 100).round(1)
    
    color = colores_fuente_modalidad[fuente]  # ← color según la fuente
    bars = ax.barh(conteos.index, pcts, color=color, height=0.4, alpha=0.9)
    
    for i, (v, p) in enumerate(zip(conteos, pcts)):
        ax.text(p + 0.3, i, f"{p}%", va="center", fontsize=9)
    
    ax.set_title(f"Modalidades más utilizadas\n{fuente.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("% de contratos")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

plt.suptitle("Distribución de modalidades de contratación por fuente", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
guardar_grafico('modalidad_pct_por_fuente')


# ============================================================
# MODALIDAD POR VALOR — separado por fuente (2 subplots)
# ============================================================

# fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# for ax, fuente in zip(axes, ["secop I", "secop II"]):
#     df_f = todo[todo["fuente"] == fuente]
#     if df_f.empty:
#         ax.text(0.5, 0.5, "Sin datos\nen este periodo", ha="center", va="center",
#                 fontsize=14, color="#999", transform=ax.transAxes)
#         ax.axis("off")
#         ax.set_title(f"Valor contratado por modalidad\n{fuente.upper()}", fontsize=13, fontweight="bold")
#         continue
#     valores = df_f.groupby("modalidad_clasificada")["precio_base"].sum().sort_values()
    
#     ax.barh(valores.index, valores.values, color="#DD8452", height=0.4)
    
#     for i, v in enumerate(valores):
#         ax.text(v, i, f" ${v/1e9:.1f}B", va="center", fontsize=9)
    
#     ax.set_title(f"Valor contratado por modalidad\n{fuente.upper()}", fontsize=13, fontweight="bold")
#     ax.set_xlabel("Valor contratado (pesos)")
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e9:.0f}B"))
#     ax.grid(axis="x", linestyle="--", alpha=0.4)
#     for spine in ["top", "right"]:
#         ax.spines[spine].set_visible(False)

# plt.suptitle("Valor de modalidades de contratación por fuente", fontsize=15, fontweight="bold", y=1.02)
# plt.tight_layout()
# guardar_grafico('modalidad_valor_por_fuente')
#INTENTO GEO
# ─────────────────────────────────────────────
# PREPARACIÓN
# ─────────────────────────────────────────────
todo["ciudad_de_la_unidad_de"] = (
    todo["ciudad_de_la_unidad_de"]
    .astype(str)
    .str.strip()
    .str.upper()
)

def formato_millones(x, pos):
    if x >= 1e12:
        return f"{x/1e12:.1f}B"
    elif x >= 1e9:
        return f"{x/1e9:.1f}MM"
    elif x >= 1e6:
        return f"{x/1e6:.1f}M"
    else:
        return f"{x:,.0f}"


# ─────────────────────────────────────────────
# FUNCIÓN PARA HACER EL GRÁFICO POR FUENTE
# ─────────────────────────────────────────────
def grafico_geo_por_fuente(df, fuente, nombre_archivo, color_barra):
    
    df_f = df[df["fuente"] == fuente]
     # ── GUARDIA ──
    if df_f.empty:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.text(0.5, 0.5, "Sin datos\nen este periodo", ha="center", va="center",
                fontsize=14, color="#999", transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"Top 10 ciudades - {fuente}", fontsize=14, fontweight="bold")
        guardar_grafico(nombre_archivo)
        return

    # contratos
    geo_contratos = (
        df_f.groupby("ciudad_de_la_unidad_de")
        .size()
        .reset_index(name="n_contratos")
    )

    # valor
    geo_valor = (
        df_f.groupby("ciudad_de_la_unidad_de")["precio_base"]
        .sum()
        .reset_index(name="valor_total")
    )

    # merge
    geo = geo_contratos.merge(geo_valor, on="ciudad_de_la_unidad_de")

    # top 10
    top_geo = geo.sort_values("valor_total", ascending=False).head(10)

    # ── GRÁFICO ──
    fig, ax1 = plt.subplots(figsize=(13,6))

    # barras valor
    ax1.barh(
        top_geo["ciudad_de_la_unidad_de"],
        top_geo["valor_total"],
        color=color_barra,
        alpha=0.9
    )

    ax1.set_xlabel("Valor total contratado")
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(formato_millones))
    ax1.invert_yaxis()

    # línea contratos
    ax2 = ax1.twiny()

    ax2.plot(
        top_geo["n_contratos"],
        top_geo["ciudad_de_la_unidad_de"],
        color="darkred",
        marker="o",
        linewidth=2
    )

    ax2.set_xlabel("Número de contratos")

    # leyenda
    legend_elements = [
        Line2D([0], [0], color=color_barra, lw=6, label="Valor contratado"),
        Line2D([0], [0], color="darkred", marker="o", lw=2, label="Número de contratos")
    ]

    ax1.legend(handles=legend_elements, loc="lower right", frameon=False)

    plt.title(f"Top 10 ciudades - {fuente}", fontsize=14, fontweight="bold")

    ax1.grid(axis="x", alpha=0.3)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    plt.tight_layout()
    guardar_grafico(nombre_archivo)


# ─────────────────────────────────────────────
# GENERAR LOS 2 GRÁFICOS
# ─────────────────────────────────────────────
grafico_geo_por_fuente(todo, "secop I",  "geo_top_secop_I",  COLOR_SECOP1)
grafico_geo_por_fuente(todo, "secop II", "geo_top_secop_II", COLOR_SECOP2)
#---------------------------------------------------------------------------------------
#ULTIMA TABLA
#---------------------------------------------------------------------------------------
cols_tabla = ["entidad", "descripci_n_del_procedimiento",
               "precio_base"]

tabla_top14 = (todo[cols_tabla]
               .assign(score=todo["score"])
               .sort_values("score", ascending=False)
               .head(14)
               .reset_index(drop=True))

#PDF ---------------------------------------------
hoy_str = date.today().strftime("%d/%m/%Y")
# ============================================================
# PDF DASHBOARD - reemplaza todo el bloque de generación PDF
# ============================================================
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread
import matplotlib.patches as mpatches

COLOR_BLUE   = "#003087"
COLOR_ACCENT = "#FFE9C5"
PDF_PATH     = os.path.join(RUTA_GRAFICOS, "informe_secop.pdf")

def fmt_peso(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/D"
    if v >= 1e12: return f"${v/1e12:.1f}B"
    if v >= 1e9:  return f"${v/1e9:.1f}MM"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def load_img(nombre):
    """Carga imagen desde RUTA_GRAFICOS, devuelve array o None."""
    path = os.path.join(RUTA_GRAFICOS, f"{nombre}.png")
    return imread(path) if os.path.exists(path) else None

def put_img(ax, nombre, title=None):
    """Pone imagen en un ax, sin ejes."""
    ax.axis("off")
    img = load_img(nombre)
    if img is not None:
        ax.imshow(img, aspect="auto", interpolation="lanczos")
        ax.set_aspect("equal")
    else:
        ax.text(0.5, 0.5, f"[{nombre}]", ha="center", va="center",
                color="gray", transform=ax.transAxes, fontsize=8)
    if title:
        ax.set_title(title, fontsize=8, color=COLOR_BLUE, fontweight="bold", pad=3)

def kpi_box(ax, label, value, color=COLOR_BLUE, fontsize_val=13):
    """Dibuja una tarjeta KPI en un ax."""
    ax.axis("off")
    ax.set_facecolor("#f0f4ff")
    rect = mpatches.FancyBboxPatch((0.05, 0.08), 0.90, 0.84,
        boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor=color, facecolor="#f0f4ff",
        transform=ax.transAxes, zorder=1)
    ax.add_patch(rect)
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=fontsize_val,
            fontweight="bold", color=color, transform=ax.transAxes, zorder=2)
    ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=7.5,
            color="#555", transform=ax.transAxes, zorder=2, multialignment="center")

def analysis_box(ax, texto):
    """Caja de texto interpretativo al pie."""
    ax.axis("off")
    ax.text(0.5, 0.5, texto, ha="center", va="center", fontsize=8.5,
            color="#333", transform=ax.transAxes, wrap=True,
            multialignment="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f7ff", ec="#c8d0e8", lw=1))

def section_header(fig, y, texto, color=COLOR_BLUE):
    """Línea de sección sobre la figura."""
    fig.text(0.5, y, texto, ha="center", fontsize=13, fontweight="bold",
             color=color)
    fig.add_artist(plt.Line2D([0.05, 0.95], [y-0.012, y-0.012],
                               color=COLOR_ACCENT, linewidth=1.5,
                               transform=fig.transFigure))

# ── Variables de análisis ─────────────────────────────────────────────────────
orden_top     = orden_contratos.sort_values("n_contratos", ascending=False).iloc[0]
orden_top_val = orden_valor.sort_values("valor_total", ascending=False).iloc[0]
modal_top_c   = modalidad_contratos.sort_values("n_contratos", ascending=False).iloc[0]
idx           = modalidad_valor["valor_total"].idxmax()
modal_top_v_nombre = modalidad_valor.loc[idx, "modalidad_clasificada"]
modal_top_v_valor  = modalidad_valor.loc[idx, "valor_total"]

if "pct_valor" not in valor_por_entidad.columns:
    valor_por_entidad["pct_valor"] = (
        valor_por_entidad["precio_base"] / valor_total * 100).round(1)
top1_entidad = valor_por_entidad.iloc[0]

quincena_pico_s1 = (contratos_quincena[contratos_quincena["fuente"] == "secop I"]
                    .sort_values("n_contratos", ascending=False).iloc[0])
quincena_pico_s2 = (contratos_quincena[contratos_quincena["fuente"] == "secop II"]
                    .sort_values("n_contratos", ascending=False).iloc[0])

# KPIs por fuente
def kpis_fuente(fuente):
    df = todo[todo["fuente"] == fuente]
    return {
        "n":      len(df),
        "valor":  df["precio_base"].sum(),
        "prom":   df["precio_base"].mean(),
        "ents":   df["entidad"].nunique(),
    }
k1 = kpis_fuente("secop I")
k2 = kpis_fuente("secop II")
MAX_CHARS_ENTIDAD = 45
top1_nombre_corto = (
    str(top1_entidad['entidad'])[:MAX_CHARS_ENTIDAD] + "…"
    if len(str(top1_entidad['entidad'])) > MAX_CHARS_ENTIDAD
    else str(top1_entidad['entidad'])
)
# ── GENERACIÓN PDF ────────────────────────────────────────────────────────────
with PdfPages(PDF_PATH) as pdf:

    # ══════════════════════════════════════════════════════
    # PÁGINA 1 — KPIs GLOBALES
    # ══════════════════════════════════════════════════════
    # fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    # fig.text(0.5, 0.93, "Informe de Contratación Pública — SECOP I & II",
    #          ha="center", fontsize=17, fontweight="bold", color=COLOR_BLUE)
    # fig.text(0.5, 0.89, f"Generado el {hoy_str}  ·  datos.gov.co",
    #          ha="center", fontsize=9, color="#888")
    # fig.add_artist(plt.Line2D([0.05,0.95],[0.875,0.875],
    #                            color=COLOR_ACCENT, lw=2,
    #                            transform=fig.transFigure))

    # kpis_globales = [
    #     ("Contratos\ntotales",    f"{total_contratos:,}"),
    #     ("Valor total",           fmt_peso(valor_total)),
    #     ("Valor promedio",        fmt_peso(valor_promedio)),
    #     ("Mediana",               fmt_peso(valor_mediana)),
    #     ("Entidades\núnicas",     f"{entidades_unicas:,}"),
    #     ("SECOP I\ncontratos",    f"{k1['n']:,}"),
    #     ("SECOP II\ncontratos",   f"{k2['n']:,}"),
    #     ("SECOP I\nvalor",        fmt_peso(k1['valor'])),
    #     ("SECOP II\nvalor",       fmt_peso(k2['valor'])),
    # ]

    # gs = GridSpec(3, 3, figure=fig,
    #               left=0.06, right=0.94, top=0.84, bottom=0.08,
    #               hspace=0.45, wspace=0.35)
    # for i, (lbl, val) in enumerate(kpis_globales):
    #     ax = fig.add_subplot(gs[i//3, i%3])
    #     color = COLOR_BLUE if i < 5 else (COLOR_ACCENT if "I\n" in lbl else "#2a9d8f")
    #     kpi_box(ax, lbl, val, color=color)

    # pdf.savefig(fig, dpi=180); plt.close(fig)

    # ══════════════════════════════════════════════════════
    # PÁGINA 2 — DASHBOARD SECOP I
    # ══════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    section_header(fig, 0.955, "SECOP I — Dashboard. Ultimas 3 semanas", color=COLOR_BLUE)
    poner_logo(fig)
    # Layout: col 0-1 = gráficos, col 2 = KPIs verticales
    gs = GridSpec(3, 3, figure=fig,
                  left=0.03, right=0.97, top=0.91, bottom=0.04,
                  hspace=0.38, wspace=0.25,
                  width_ratios=[4, 4, 1])

    # fila 0: torta contratos | torta valor | KPIs
    ax00 = fig.add_subplot(gs[0, 0]); put_img(ax00, "torta_orden_secop1", "Contratos por tipo entidad")
    ax01 = fig.add_subplot(gs[0, 1]); put_img(ax01, "torta_valor_secop1", "Valor por tipo entidad")
    ax02 = fig.add_subplot(gs[0, 2]); kpi_box(ax02, "Contratos\nSECOP I", f"{k1['n']:,}", COLOR_BLUE)

    # fila 1: top10 | geo | más KPIs
    ax10 = fig.add_subplot(gs[1, 0]); put_img(ax10, "top10_secop_I", "Top 10 entidades")
    ax11 = fig.add_subplot(gs[1, 1]); put_img(ax11, "geo_top_secop_I", "Top ciudades")
    ax12 = fig.add_subplot(gs[1, 2]); kpi_box(ax12, "Valor total\nSECOP I", fmt_peso(k1['valor']), COLOR_BLUE)

    # fila 2: top 5/10 | análisis | KPI
    ax20 = fig.add_subplot(gs[2, 0]); put_img(ax20, "top_5_10_secop_I", "Concentración valor")
    ax21 = fig.add_subplot(gs[2, 1])
    analysis_box(ax21,
        f"SECOP I registra {k1['n']:,} contratos por {fmt_peso(k1['valor'])}.\n"
        f"El tipo de entidad dominante es '{orden_top['orden_clasificado']}' "
        f"({orden_top['n_contratos']/total_contratos*100:.1f}% del total).\n"
        f"Las 5 principales entidades concentran el {top5}% del valor.\n"
        f"Pico de actividad: quincena '{quincena_pico_s1['quincena_str']}' "
        f"con {int(quincena_pico_s1['n_contratos'])} contratos."
    )
    ax22 = fig.add_subplot(gs[2, 2]); kpi_box(ax22, "Entidades\núnicas I", f"{k1['ents']:,}", COLOR_BLUE)

    pdf.savefig(fig, dpi=180); plt.close(fig)

    # ══════════════════════════════════════════════════════
    # PÁGINA 3 — DASHBOARD SECOP II
    # ══════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    section_header(fig, 0.955, "SECOP II — Dashboard", color='#D8841C')
    poner_logo(fig)
    gs = GridSpec(3, 3, figure=fig,
                  left=0.03, right=0.97, top=0.91, bottom=0.04,
                  hspace=0.38, wspace=0.25,
                  width_ratios=[4, 4, 1])

    ax00 = fig.add_subplot(gs[0, 0]); put_img(ax00, "torta_orden_secop2", "Contratos por tipo entidad")
    ax01 = fig.add_subplot(gs[0, 1]); put_img(ax01, "torta_valor_secop2", "Valor por tipo entidad")
    ax02 = fig.add_subplot(gs[0, 2]); kpi_box(ax02, "Contratos\nSECOP II", f"{k2['n']:,}", "#2a9d8f")

    ax10 = fig.add_subplot(gs[1, 0]); put_img(ax10, "top10_secop_II", "Top 10 entidades")
    ax11 = fig.add_subplot(gs[1, 1]); put_img(ax11, "geo_top_secop_II", "Top ciudades")
    ax12 = fig.add_subplot(gs[1, 2]); kpi_box(ax12, "Valor total\nSECOP II", fmt_peso(k2['valor']), "#2a9d8f")

    ax20 = fig.add_subplot(gs[2, 0]); put_img(ax20, "top_5_10_secop_II", "Concentración valor")
    ax21 = fig.add_subplot(gs[2, 1])
    analysis_box(ax21,
        f"SECOP II registra {k2['n']:,} contratos por {fmt_peso(k2['valor'])}.\n"
        f"La entidad con mayor peso es '{top1_nombre_corto}' "
        f"con {top1_entidad['pct_valor']:.1f}% del valor total.\n"
        f"Las 10 principales entidades concentran el {top10}% del valor.\n"
        f"Pico de actividad: quincena '{quincena_pico_s2['quincena_str']}' "
        f"con {int(quincena_pico_s2['n_contratos'])} contratos."
    )
    ax22 = fig.add_subplot(gs[2, 2]); kpi_box(ax22, "Entidades\núnicas II", f"{k2['ents']:,}", "#2a9d8f")

    pdf.savefig(fig, dpi=180); plt.close(fig)

    # ══════════════════════════════════════════════════════
    # PÁGINA 4 — MODALIDADES
    # ══════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    section_header(fig, 0.955, "Modalidades de Contratación — Tabla busqueda semanticaa")
    poner_logo(fig)
    gs = GridSpec(3, 1, figure=fig,
              left=0.03, right=0.97, top=0.91, bottom=0.03,
              hspace=0.3,
              height_ratios=[3, 2.5, 0.6])
    ax0 = fig.add_subplot(gs[0])
    put_img(ax0, "modalidad_pct_por_fuente", "% de contratos por modalidad")

    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    ax1.set_title("Top contratos por relevancia semántica", fontsize=10, fontweight="bold", color=COLOR_BLUE)

    data = []
    for _, row in tabla_top14.iterrows():
        data.append([
            str(row["entidad"])[:30],
            str(row["descripci_n_del_procedimiento"])[:60],
            fmt_peso(row["precio_base"]),
        ])

    tbl = ax1.table(          # ← ax1, no ax
        cellText=data,
        colLabels=["Entidad", "Descripción", "Valor"],   # ← 3 columnas
        cellLoc="left", loc="center",
        colWidths=[0.30, 0.50, 0.20],                    # ← 3 anchos
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(COLOR_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        cell.set_edgecolor("#dde")

    ax1.text(0.5, 0.02, "Ordenado por similitud semántica con las palabras clave de búsqueda.",
            ha="center", fontsize=8, color="#999", transform=ax1.transAxes)  # ← ax1
    # ax1 = fig.add_subplot(gs[1]); put_img(ax1, "modalidad_valor_por_fuente", "Valor contratado por modalidad") aca puedo poner la tabla
    ax2 = fig.add_subplot(gs[2])
    analysis_box(ax2,
        f"En SECOP I, '{modal_top_c['modalidad_clasificada']}' lidera en frecuencia "
        f"({modal_top_c['n_contratos']:,} contratos, {modal_top_c['pct']:.1f}%). "
        f"En valor, '{modal_top_v_nombre}' acumula {fmt_peso(modal_top_v_valor)}. "
        f"SECOP II concentra sus contratos en Menor Cuantía (100% de sus registros)."
    )


    pdf.savefig(fig, dpi=180); plt.close(fig)

    # ══════════════════════════════════════════════════════
    # PÁGINA 5 — SERIE DE TIEMPO
    # ══════════════════════════════════════════════════════
    # fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    # section_header(fig, 0.955, "Evolución Quincenal de Contratos")

    # gs = GridSpec(2, 1, figure=fig,
    #               left=0.05, right=0.95, top=0.91, bottom=0.10,
    #               height_ratios=[5, 1], hspace=0.2)

    # ax0 = fig.add_subplot(gs[0]); put_img(ax0, "serie_tiempo")
    # ax1 = fig.add_subplot(gs[1])
    # analysis_box(ax1,
    #     f"SECOP I alcanzó su pico en '{quincena_pico_s1['quincena_str']}' "
    #     f"con {int(quincena_pico_s1['n_contratos'])} contratos. "
    #     f"SECOP II tuvo su mayor actividad en '{quincena_pico_s2['quincena_str']}' "
    #     f"con {int(quincena_pico_s2['n_contratos'])} contratos. "
    #     f"La caída al final del periodo puede reflejar datos aún incompletos por publicación reciente."
    # )

    # pdf.savefig(fig, dpi=180); plt.close(fig)

    # # ══════════════════════════════════════════════════════
    # # PÁGINA 6 — TABLA TOP 14
    # # ══════════════════════════════════════════════════════
    # fig, ax = plt.subplots(figsize=(17, 8))
    # fig.patch.set_facecolor("white")
    # poner_logo(fig)
    # ax.axis("off")
    # ax.set_title("Top 14 contratos por relevancia semántica",
    #              fontsize=14, fontweight="bold", color=COLOR_BLUE, pad=12)

    # def get_url(val):
    #     if pd.isna(val) if not isinstance(val, dict) else False: return "N/D"
    #     if isinstance(val, dict): return str(list(val.values())[0])
    #     return str(val)

    # data = []
    # for _, row in tabla_top14.iterrows():
    #     data.append([
    #         str(row["entidad"])[:30],
    #         str(row["descripci_n_del_procedimiento"])[:60],
    #         # str(row["modalidad_clasificada"])[:21],
    #         # str(row["fase"])[:12],
    #         fmt_peso(row["precio_base"]),
    #         # str(row["fuente"]),
    #         # get_url(row["urlproceso"])
    #     ])

    # tbl = ax.table(
    #     cellText=data,
    #     colLabels=["Entidad","Descripción","Valor","Fuente","URL"],
    #     cellLoc="left", loc="center",
    #     colWidths=[0.20, 0.38, 0.15, 0.10, 0.10, 0.08, 0.38],
    # )
    # tbl.auto_set_font_size(False)
    # tbl.set_fontsize(7.5)
    # tbl.scale(1, 1.6)

    # for (r, c), cell in tbl.get_celld().items():
    #     if r == 0:
    #         cell.set_facecolor(COLOR_BLUE)
    #         cell.set_text_props(color="white", fontweight="bold")
    #     elif r % 2 == 0:
    #         cell.set_facecolor("#f0f4ff")
    #     cell.set_edgecolor("#dde")

    # ax.text(0.5, 0.02, "Ordenado por similitud semántica con las palabras clave de búsqueda.",
    #         ha="center", fontsize=8, color="#999", transform=ax.transAxes)

    # pdf.savefig(fig, bbox_inches="tight", dpi=180); plt.close(fig)

print(f"✅ PDF generado: {PDF_PATH}")
# ── PASO 5: Enviar correo ─────────────────────────────────────────────────────
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_TO_RAW   = os.environ.get("EMAIL_TO")
EMAIL_TO       = [e.strip() for e in EMAIL_TO_RAW.split(",") if e.strip()]
SMTP_HOST      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.environ.get("SMTP_PORT", 587))
hoy_legible    = date.today().strftime("%d de %B de %Y")
fuentes_resumen = "  |  ".join([
    f"{f}: {n:,}" for f, n in todo["fuente"].value_counts().items()
])

msg            = MIMEMultipart()
msg["Subject"] = f"Informe SECOP Javeriana - {hoy_legible} - {total_contratos:,} contratos"
msg["From"]    = EMAIL_SENDER
msg["To"]      = ", ".join(EMAIL_TO)

cuerpo = (
    f"Hola,\n\n"
    f"Adjunto el informe de contratos SECOP relacionados con.\n\n"
    f"Palabras clave utilizadas: {', '.join(palabras_clave)}.\n\n"
    f"Resumen:\n"
    f"   - Contratos encontrados : {total_contratos:,}\n"
    f"   - {fuentes_resumen}\n"
    f"   - Monto total           : {fmt_peso(valor_total)}\n"
    f"   - Monto promedio        : {fmt_peso(valor_promedio)}\n"
    f"   - Mediana               : {fmt_peso(valor_mediana)}\n"
    f"   - Entidades únicas      : {entidades_unicas:,}\n"
    f"   - Top 5 entidades       : {top5}% del valor total\n"
    f"   - Top 10 entidades      : {top10}% del valor total\n\n"
    f"Fuente: datos.gov.co - SECOP I (Convocado) + SECOP II (Publicado)\n"
    f"Generado automaticamente el {hoy_legible}.\n"
)
msg.attach(MIMEText(cuerpo, "plain"))

with open(PDF_PATH, "rb") as f:
    part = MIMEBase("application", "octet-stream")
    part.set_payload(f.read())
encoders.encode_base64(part)
part.add_header("Content-Disposition",
                f"attachment; filename=informe_secop_{date.today()}.pdf")
msg.attach(part)

try:
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_TO, msg.as_string())
    print(f"Correo enviado a: {', '.join(EMAIL_TO)}")
except Exception as e:
    print(f"Error al enviar correo: {e}")
    print(f"PDF guardado igualmente en: {PDF_PATH}")

print("\nProceso finalizado correctamente.")

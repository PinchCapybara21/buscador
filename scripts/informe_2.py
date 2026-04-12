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

RUTA_GRAFICOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(RUTA_GRAFICOS, exist_ok=True)

def guardar_grafico(nombre):
    ruta = os.path.join(RUTA_GRAFICOS, f"{nombre}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
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
    "modalidad_de_contratacion", "fase",'ciudad_de_la_unidad_de', 'departamento_entidad',
    "descripci_n_del_procedimiento", "fecha_de_publicacion_del",
    "precio_base",
    "urlproceso", "score"
]
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
todo = pd.concat([resultados_secop1, resultados_secop2], ignore_index=True)
# print(todo.to_string(max_colwidth=None, index=False))
"""
# =========================================
# 🧪 PRUEBA
# =========================================
consulta = "cundinamarca"
resultados = buscar_semantico(consulta, top_k=5)

from tabulate import tabulate

print(f"\n🔎 Resultados para: {consulta}\n")
print(tabulate(resultados, headers='keys', tablefmt='pretty', showindex=False))
"""
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

"""
# Proveedores únicos — activar cuando tengas la columna
proveedores_unicos = todo["nombre_proveedor"].nunique()
"""

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
    if v in ["TERRITORIAL DEPARTAMENTAL CENTRALIZADO", "TERRITORIAL DEPARTAMENTAL DESCENTRALIZADO"]:
        return "Territorial Departamental"
    if v in ["TERRITORIAL DISTRITAL MUNICIPAL NIVEL 2", "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 4",
             "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 5", "TERRITORIAL DISTRITAL MUNICIPAL NIVEL 6"]:
        return "Territorial Municipal"
    if v == "Territorial":
        return "Territorial"
    if v == "Corporación Autónoma":
        return "Corporación Autónoma"
    return "No Definido"

todo["orden_clasificado"] = todo["ordenentidad"].apply(clasificar_orden)

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
    v = str(valor).upper()
    if "DIRECTA" in v:
        return "Contratación Directa"
    if "LICITACi" in v or "LICITACI" in v:
        return "Licitación Pública"
    if "MENOR CUANT" in v:
        return "Menor Cuantía"
    if "CONCURSO" in v:
        return "Concurso de Méritos"
    if "SUBASTA" in v:
        return "Subasta"
    if "CONVENIOS" in v or "PARTES" in v:
        return "Convenios"
    if "ESPECIAL" in v:
        return "Régimen Especial"
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

conteo = todo["orden_clasificado"].value_counts()

conteo = conteo.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))

def formato(pct):
    total = conteo.sum()
    val = int(round(pct * total / 100))
    return f"{pct:.1f}%\n({val})"
wedges, _ = ax.pie(
    conteo,
    startangle=90,
    wedgeprops=dict(width=0.38),)
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

ax.set_title("Distribución de contratos por tipo de entidad",
             fontsize=14, weight='semibold', pad=15)

plt.subplots_adjust(right=0.68)
guardar_grafico('tora_1')
#tora 2 _---------------------------------------------------------------------------------------------------
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
ax.set_title("Distribución de precio base por tipo de entidad",
             fontsize=14, weight="semibold", pad=15)

plt.subplots_adjust(right=0.65)
plt.tight_layout()
guardar_grafico('torta_2')
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

pivot = contratos_quincena.pivot(
    index="quincena_str",
    columns="fuente",
    values="n_contratos"
).fillna(0)

pivot = pivot.sort_index()

# =========================================
# 🎨 GRÁFICO
# =========================================
plt.figure(figsize=(13,6))

line1, = plt.plot(pivot.index, pivot["secop I"], marker='o', linewidth=2)
line2, = plt.plot(pivot.index, pivot["secop II"], marker='s', linewidth=2, linestyle='--')

# =========================================
# ✨ ESTÉTICA
# =========================================
plt.title("Evolución quincenal de contratos", fontsize=16, fontweight='bold')
plt.xlabel("Periodo (Quincena)")
plt.ylabel("Número de contratos")

plt.xticks(rotation=45)
plt.grid(alpha=0.3)

# Quitar bordes feos
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

# =========================================
# 📌 LEYENDA AL LADO (CLAVE)
# =========================================
plt.legend(
    [line1, line2],
    ["SECOP I", "SECOP II"],
    title="Fuente",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    frameon=False
)

plt.tight_layout()
guardar_grafico('serie_tiempo')

# TOP 5 Y 10 -------------------------------------------------------------------------------------------------------
top5_entidades  = valor_por_entidad.head(5)["entidad"]
top10_entidades = valor_por_entidad.head(10)["entidad"]

# Concentración por fuente SECOP
def secop_concentracion(entidades):
    df = todo[todo["entidad"].isin(entidades)]
    return df.groupby("fuente")["precio_base"].sum()

top5_secop_pct  = (secop_concentracion(top5_entidades)  / valor_total * 100).round(1)
top10_secop_pct = (secop_concentracion(top10_entidades) / valor_total * 100).round(1)

# Grafico
labels    = ["Top 5", "Top 10"]
secop_i   = [top5_secop_pct.get("secop I", 0),  top10_secop_pct.get("secop I", 0)]
secop_ii  = [top5_secop_pct.get("secop II", 0), top10_secop_pct.get("secop II", 0)]

plt.figure(figsize=(7, 5))
plt.bar(labels, secop_i,  label="SECOP I")
plt.bar(labels, secop_ii, bottom=secop_i, label="SECOP II")

plt.title("Concentración del valor contratado por SECOP", fontsize=14, fontweight="bold")
plt.ylabel("% del valor total")
plt.ylim(0, 100)

for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.grid(axis="y", alpha=0.3)
plt.legend()

totales = [a + b for a, b in zip(secop_i, secop_ii)]
for i, total in enumerate(totales):
    plt.text(i, total + 1, f"{total:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
guardar_grafico('top_5_10')

top_n = 10

valor_por_entidad = (
    todo.groupby("entidad")["precio_base"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

top_entidades = valor_por_entidad.head(top_n).copy()

top_entidades["pct_valor"] = (
    top_entidades["precio_base"] / todo["precio_base"].sum() * 100
)


df_top = todo[todo["entidad"].isin(top_entidades["entidad"])]


tabla = (
    df_top.groupby(["entidad", "fuente"])["precio_base"]
    .sum()
    .unstack()
    .fillna(0)
)

tabla = tabla.reindex(top_entidades["entidad"])


total_general = todo["precio_base"].sum()
tabla_pct = tabla / total_general * 100

fig, ax = plt.subplots(figsize=(12,7))

ax.barh(
    top_entidades["entidad"],
    tabla_pct["secop I"],
    label="SECOP I"
)

ax.barh(
    top_entidades["entidad"],
    tabla_pct["secop II"],
    left=tabla_pct["secop I"],
    label="SECOP II"
)

ax.invert_yaxis()

ax.set_title("Descomposición del peso de cada entidad por SECOP", fontsize=14)
ax.set_xlabel("% del total del sistema")

totales_entidad = top_entidades["pct_valor"]

for i, total in enumerate(totales_entidad):
    ax.text(
        total + 0.3,
        i,
        f"{total:.1f}%",
        va="center",
        fontweight="bold"
    )

ax.grid(axis="x", alpha=0.3)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend()

plt.tight_layout()
guardar_grafico('top_valor')
#MODALIDAD DE CONTRATACION--------------------------------------------------------------------------------------------------------
total = modalidad_contratos["n_contratos"].sum()
modalidad_contratos["pct"] = (modalidad_contratos["n_contratos"] / total * 100).round(1)

tabla_secop = (
    todo.groupby(["modalidad_clasificada", "fuente"])
    .size()
    .unstack()
    .fillna(0)
)

# asegurar columnas
for col in ["secop I", "secop II"]:
    if col not in tabla_secop.columns:
        tabla_secop[col] = 0

tabla_secop = tabla_secop[["secop I", "secop II"]]

# ordenar igual que el gráfico principal
tabla_secop = tabla_secop.reindex(modalidad_contratos["modalidad_clasificada"])

# ================================
# 5. GRÁFICO
# ================================
fig, ax = plt.subplots(figsize=(10,6))

ax.barh(tabla_secop.index, tabla_secop["secop I"], label="SECOP I")
ax.barh(tabla_secop.index, tabla_secop["secop II"], left=tabla_secop["secop I"], label="SECOP II")

ax.set_title("Modalidades más utilizadas y su distribución SECOP I vs II", fontsize=14, fontweight="bold")
ax.set_xlabel("Número de contratos")

# ================================
# 6. ETIQUETAS (% de uso de modalidad)
# ================================
for i, (v, p) in enumerate(zip(modalidad_contratos["n_contratos"], modalidad_contratos["pct"])):
    ax.text(v + 1, i, f"{p}%", va="center")

ax.grid(axis="x", linestyle="--", alpha=0.4)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend()

plt.tight_layout()
guardar_grafico('modalidada')
# MODALIDAD VALOOORRR
modalidad_valor = (
    todo.groupby(["modalidad_clasificada", "fuente"])["precio_base"]
    .sum()
    .unstack()
    .fillna(0)
)
for col in ["secop I", "secop II"]:
    if col not in modalidad_valor.columns:
        modalidad_valor[col] = 0

modalidad_valor = modalidad_valor[["secop I", "secop II"]]

modalidad_valor = modalidad_valor.loc[modalidad_valor.sum(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(modalidad_valor.index, modalidad_valor["secop I"], label="SECOP I")
ax.barh(modalidad_valor.index, modalidad_valor["secop II"], left=modalidad_valor["secop I"], label="SECOP II")

ax.set_title("Modalidades de contratación por valor (SECOP I vs II)", fontsize=14, fontweight="bold")
ax.set_xlabel("Valor contratado")

totales = modalidad_valor["secop I"] + modalidad_valor["secop II"]

for i, total in enumerate(totales):
    ax.text(total, i, f"{total:,.0f}", va="center", fontweight="bold")

ax.grid(axis="x", alpha=0.3)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend()

plt.tight_layout()
guardar_grafico('modalidad_valor')
#INTENTO GEO
todo["ciudad_de_la_unidad_de"] = todo["ciudad_de_la_unidad_de"].astype(str).str.strip().str.upper()

geo_contratos = todo.groupby("ciudad_de_la_unidad_de").size().reset_index(name="n_contratos")

geo_secop = todo.groupby(["ciudad_de_la_unidad_de", "fuente"])["precio_base"].sum().unstack().fillna(0)

geo = geo_contratos.merge(geo_secop, on="ciudad_de_la_unidad_de")

geo["valor_total"] = geo.get("secop I", 0) + geo.get("secop II", 0)

top_geo = geo.sort_values("valor_total", ascending=False).head(10)

def formato_millones(x, pos):
    if x >= 1e12:
        return f"{x/1e12:.1f}B"
    elif x >= 1e9:
        return f"{x/1e9:.1f}MM"
    elif x >= 1e6:
        return f"{x/1e6:.1f}M"
    else:
        return f"{x:,.0f}"

fig, ax1 = plt.subplots(figsize=(13,6))

ax1.barh(
    top_geo["ciudad_de_la_unidad_de"],
    top_geo.get("secop I", 0),
    color="#4C78A8",
    label="SECOP I"
)

ax1.barh(
    top_geo["ciudad_de_la_unidad_de"],
    top_geo.get("secop II", 0),
    left=top_geo.get("secop I", 0),
    color="#F58518",
    label="SECOP II"
)

ax1.set_xlabel("Valor total contratado")
ax1.xaxis.set_major_formatter(mtick.FuncFormatter(formato_millones))
ax1.invert_yaxis()

ax2 = ax1.twiny()

ax2.plot(
    top_geo["n_contratos"],
    top_geo["ciudad_de_la_unidad_de"],
    color="darkred",
    marker="o",
    linewidth=2,
    label="Contratos"
)

ax2.set_xlabel("Número de contratos")

legend_elements = [
    Line2D([0], [0], color="#4C78A8", lw=6, label="SECOP I"),
    Line2D([0], [0], color="#F58518", lw=6, label="SECOP II"),
    Line2D([0], [0], color="darkred", marker="o", lw=2, label="Número de contratos")
]

ax1.legend(handles=legend_elements, loc="lower right", frameon=False)

plt.title("Top 10 ciudades: SECOP I vs II + contratos", fontsize=14, fontweight="bold")

ax1.grid(axis="x", alpha=0.3)

for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

plt.tight_layout()
guardar_grafico('geo_top')
#PDF ---------------------------------------------
hoy_str = date.today().strftime("%d/%m/%Y")
# ── GENERACIÓN DE PDF ────────────────────────────────────────────────────────
# Pega este bloque al final de tu script principal.
# Usa ÚNICAMENTE variables ya definidas: todo, valor_total, total_contratos,
# valor_promedio, valor_mediana, entidades_unicas, top5, top10,
# orden_contratos, orden_valor, contratos_quincena, modalidad_contratos,
# modalidad_valor, valor_por_entidad, hoy_str, RUTA_GRAFICOS

COLOR_BLUE = "#003087"
PDF_PATH   = os.path.join(RUTA_GRAFICOS, "informe_secop.pdf")

def fmt_peso(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/D"
    if v >= 1e12:  return f"${v/1e12:.1f}B"
    if v >= 1e9:   return f"${v/1e9:.1f}MM"
    if v >= 1e6:   return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def texto_wrap(ax, texto, y=0.5, fontsize=10):
    ax.text(0.5, y, texto, ha="center", va="center", fontsize=fontsize,
            color="#333", transform=ax.transAxes, wrap=True,
            multialignment="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f7ff", ec="#c8d0e8", lw=1))

def agregar_pagina_grafico(pdf, img_path, titulo, texto):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5),
                              gridspec_kw={"height_ratios": [4, 1]})
    # Gráfico
    ax_img = axes[0]
    ax_img.axis("off")
    if os.path.exists(img_path):
        from matplotlib.image import imread
        img = imread(img_path)
        ax_img.imshow(img, aspect="auto")
    else:
        ax_img.text(0.5, 0.5, "Imagen no encontrada", ha="center", va="center",
                    transform=ax_img.transAxes, color="red")
    ax_img.set_title(titulo, fontsize=14, fontweight="bold", color=COLOR_BLUE, pad=8)

    # Texto interpretativo
    ax_txt = axes[1]
    ax_txt.axis("off")
    texto_wrap(ax_txt, texto, y=0.5, fontsize=9.5)

    fig.tight_layout(pad=1.5)
    pdf.savefig(fig)
    plt.close(fig)

# ── Textos automáticos ───────────────────────────────────────────────────────
orden_top     = orden_contratos.sort_values("n_contratos", ascending=False).iloc[0]
orden_top_val = orden_valor.sort_values("valor_total", ascending=False).iloc[0]
modal_top_c   = modalidad_contratos.sort_values("n_contratos", ascending=False).iloc[0]
modal_top_v_nombre = modalidad_valor.sum(axis=1).idxmax()
modal_top_v_valor  = modalidad_valor.sum(axis=1).max()
top1_entidad  = valor_por_entidad.iloc[0]
if "pct_valor" not in valor_por_entidad.columns:
    valor_por_entidad["pct_valor"] = (valor_por_entidad["precio_base"] / valor_total * 100).round(1)
    top1_entidad = valor_por_entidad.iloc[0]

quincena_pico_s1 = (contratos_quincena[contratos_quincena["fuente"] == "secop I"]
                    .sort_values("n_contratos", ascending=False).iloc[0])
quincena_pico_s2 = (contratos_quincena[contratos_quincena["fuente"] == "secop II"]
                    .sort_values("n_contratos", ascending=False).iloc[0])

textos = {
    "tora_1": (
        f"De los {total_contratos:,} contratos identificados, la categoría con mayor participación "
        f"es '{orden_top['orden_clasificado']}' con {orden_top['n_contratos']:,} contratos "
        f"({orden_top['n_contratos']/total_contratos*100:.1f}%). La distribución refleja el nivel de "
        f"gobierno que más ha contratado en los temas buscados."
    ),
    "torta_2": (
        f"En términos de valor, '{orden_top_val['orden_clasificado']}' concentra la mayor proporción "
        f"del presupuesto con {fmt_peso(orden_top_val['valor_total'])} sobre un total de "
        f"{fmt_peso(valor_total)}. Esto puede diferir del conteo de contratos, lo que indica "
        f"presencia de contratos de alto valor en ciertos niveles de gobierno."
    ),
    "serie_tiempo": (
        f"La evolución quincenal muestra que SECOP I alcanzó su pico en la quincena "
        f"'{quincena_pico_s1['quincena_str']}' con {int(quincena_pico_s1['n_contratos'])} contratos, "
        f"mientras SECOP II tuvo su mayor actividad en '{quincena_pico_s2['quincena_str']}' "
        f"con {int(quincena_pico_s2['n_contratos'])} contratos. La tendencia al final del periodo "
        f"puede reflejar datos aún incompletos por publicación reciente."
    ),
    "top_5_10": (
        f"Las 5 entidades con mayor gasto concentran el {top5}% del valor total contratado, "
        f"y las 10 primeras acumulan el {top10}%. Este nivel de concentración sugiere que un "
        f"grupo reducido de entidades domina la contratación en los temas analizados."
    ),
    "top_valor": (
        f"La entidad con mayor peso es '{top1_entidad['entidad']}' con el "
        f"{top1_entidad['pct_valor']:.1f}% del valor total ({fmt_peso(top1_entidad['precio_base'])}). "
        f"El color indica si el contrato proviene de SECOP I o II, permitiendo identificar "
        f"qué plataforma usa cada entidad."
    ),
    "modalidad_valor": (
        f"La modalidad '{modal_top_v_nombre}' es la de mayor valor contratado "
        f"({fmt_peso(modal_top_v_valor)}), mientras '{modal_top_c['modalidad_clasificada']}' "
        f"es la más frecuente en número de contratos ({modal_top_c['n_contratos']:,}). "
        f"Una brecha grande entre ambos rankings puede indicar contratos atípicamente grandes."
    ),
    "modalidada": (
    f"En número de contratos, '{modal_top_c['modalidad_clasificada']}' es la modalidad más usada "
    f"({modal_top_c['n_contratos']:,} contratos, {modal_top_c['pct']:.1f}%), seguida de Contratación Directa. "
    f"SECOP I predomina en Régimen Especial y Contratación Directa, mientras SECOP II concentra "
    f"los contratos de Menor Cuantía, lo que refleja los procesos que cada plataforma gestiona por defecto."
    ),
    "geo_top": (
    f"Las 10 ciudades con mayor valor contratado están encabezadas por "
    f"'{top_geo.iloc[0]['ciudad_de_la_unidad_de']}' con {fmt_peso(top_geo.iloc[0]['valor_total'])}, "
    f"seguida de '{top_geo.iloc[1]['ciudad_de_la_unidad_de']}' con {fmt_peso(top_geo.iloc[1]['valor_total'])}. "
    f"La línea roja muestra que la ciudad con más contratos es "
    f"'{top_geo.sort_values('n_contratos', ascending=False).iloc[0]['ciudad_de_la_unidad_de']}' "
    f"({int(top_geo['n_contratos'].max())} contratos), lo que indica que alto valor y alta frecuencia "
    f"no siempre coinciden en la misma ciudad."
)
}

# ── Generar PDF ───────────────────────────────────────────────────────────────
graficos = [
    ("tora_1",        "Distribución de contratos por tipo de entidad"),
    ("torta_2",       "Distribución de precio base por tipo de entidad"),
    ("serie_tiempo",  "Evolución quincenal de contratos"),
    ("top_5_10",      "Concentración del valor contratado — Top 5 y Top 10"),
    ("top_valor",     "Descomposición del valor por entidad"),
    ("modalidad_valor","Modalidades de contratación por valor"),
    ("modalidada", "Modalidades más utilizadas y su distribución SECOP I vs II"),
    ("geo_top",       "Top 10 ciudades por valor contratado y número de contratos")
]

with PdfPages(PDF_PATH) as pdf:

    # ── PORTADA ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(COLOR_BLUE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor(COLOR_BLUE)

    # Franja lateral derecha con íconos (simulada con rectángulos de colores)
    colores_franja = ["#e8501a", "#f4a020", "#2a9d8f", "#264653", "#e76f51", "#2a9d8f", "#e8501a", "#f4a020", "#2a9d8f"]
    for i, c in enumerate(colores_franja):
        rect = plt.Rectangle((0.88, 1 - (i+1)*0.111), 0.12, 0.105,
                              transform=ax.transAxes, color=c, zorder=2)
        ax.add_patch(rect)

    # Franja naranja superior
    rect_top = plt.Rectangle((0, 0.88), 0.87, 0.12,
                              transform=ax.transAxes, color="#e8501a", zorder=2)
    ax.add_patch(rect_top)

    # Número de informe
    ax.text(0.04, 0.935, f"INFORME SECOP · {hoy_str}", color="white",
            fontsize=9, transform=ax.transAxes, zorder=3)

    # Título principal
    ax.text(0.06, 0.72, "INFORME SOBRE\nCONTRATACIÓN\nPÚBLICA EN\nMEDIOS Y\nMARKETING",
            color="white", fontsize=28, fontweight="bold",
            transform=ax.transAxes, va="top", linespacing=1.3, zorder=3)

    # Subtítulo
    ax.text(0.06, 0.40,
            f"Informe semántico de contratos en SECOP I y II\n"
            f"relacionados con publicidad, marketing y medios de comunicación.",
            color="#dddddd", fontsize=10, transform=ax.transAxes,
            va="top", linespacing=1.5, zorder=3)

    # Línea separadora
    ax.axhline(y=0.22, xmin=0.06, xmax=0.86, color="#e8501a", linewidth=1.5)

    # Autores
    ax.text(0.06, 0.20, "Juan Pablo Perilla Silva\nLiceth Daniela Mora Vanegas",
            color="white", fontsize=10, transform=ax.transAxes, va="top",
            linespacing=1.6, zorder=3)

    # Supervisión
    ax.text(0.06, 0.11, "Supervisado por: Carlos J. Ortiz Bonilla",
            color="#aac8ff", fontsize=9, transform=ax.transAxes, zorder=3)

    # Logo texto inferior
    ax.text(0.06, 0.04, "Observatorio Fiscal · Pontificia Universidad Javeriana",
            color="#aac8ff", fontsize=8, transform=ax.transAxes, zorder=3)

    pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_facecolor("white"); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.95, "Indicadores clave", ha="center", fontsize=18,
            fontweight="bold", color=COLOR_BLUE, transform=ax.transAxes)
    ax.plot([0.05, 0.95], [0.925, 0.925], color="#e8501a", linewidth=2,
            transform=ax.transAxes, zorder=3)

    kpis = [
        ("Contratos\nencontrados",  f"{total_contratos:,}"),
        ("Valor total",             fmt_peso(valor_total)),
        ("Valor promedio",          fmt_peso(valor_promedio)),
        ("Mediana",                 fmt_peso(valor_mediana)),
        ("Entidades únicas",        f"{entidades_unicas:,}"),
        ("SECOP I\n(contratos)",    f"{part.loc['secop I','n_contratos']:,}"
                                    f"\n{part.loc['secop I','pct_contratos']}%"),
        ("SECOP II\n(contratos)",   f"{part.loc['secop II','n_contratos']:,}"
                                    f"\n{part.loc['secop II','pct_contratos']}%"),
        ("SECOP I\n(valor)",        f"{fmt_peso(part.loc['secop I','valor'])}"
                                    f"\n{part.loc['secop I','pct_valor']}%"),
        ("SECOP II\n(valor)",       f"{fmt_peso(part.loc['secop II','valor'])}"
                                    f"\n{part.loc['secop II','pct_valor']}%"),
    ]

    cols, rows = 3, 3
    card_w, card_h = 0.27, 0.20
    x_starts = [0.05, 0.37, 0.69]
    y_starts  = [0.68, 0.42, 0.16]

    for idx, (lbl, val) in enumerate(kpis):
        col = idx % cols
        row = idx // cols
        x, y = x_starts[col], y_starts[row]

        rect = plt.Rectangle((x, y), card_w, card_h,
                              transform=ax.transAxes,
                              facecolor="#f0f4ff", edgecolor=COLOR_BLUE,
                              linewidth=1.5, zorder=2)
        ax.add_patch(rect)

        ax.text(x + card_w/2, y + card_h*0.63, val,
                ha="center", va="center", fontsize=15, fontweight="bold",
                color=COLOR_BLUE, transform=ax.transAxes, zorder=3)
        ax.text(x + card_w/2, y + card_h*0.22, lbl,
                ha="center", va="center", fontsize=8.5, color="#555",
                transform=ax.transAxes, zorder=3)

    ax.text(0.5, 0.06,
            f"Fuente: SECOP I (Convocado) + SECOP II (Publicado) · datos.gov.co · {hoy_str}",
            ha="center", fontsize=8, color="#999", transform=ax.transAxes)

    pdf.savefig(fig); plt.close(fig)
    # Páginas de gráficos
    for nombre, titulo in graficos:
        img_path = os.path.join(RUTA_GRAFICOS, f"{nombre}.png")
        agregar_pagina_grafico(pdf, img_path, titulo, textos[nombre])

print(f"✅ PDF generado: {PDF_PATH}")
# ── PASO 5: Enviar correo ─────────────────────────────────────────────────────
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER",   "almacenbiancasilva11@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "yylxvoxojnjlvfae")
EMAIL_TO_RAW   = os.environ.get("EMAIL_TO",       "juansilva11@outlook.com, danimorav05@gmail.com, hannagabrielahidalgorodriguez@gmail.com")
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
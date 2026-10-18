"""Servidor MCP local para consultas sobre reseñas de Amazon. Iteración V3.2: filtro por estrellas en búsqueda.

Requisitos: haber ejecutado 01_preparar_resenas.py y 02_construir_indice.py.
Levantar: python 04_servidor_mcp.py
Endpoint: http://127.0.0.1:8000/mcp

Este servidor contiene exactamente dos tools MCP:
    1. buscar_resenas: búsqueda semántica con EmbeddingGemma + FAISS.
    2. estadisticas_resenas: estadísticas exactas de la muestra mediante pandas.

El corpus fue seleccionado con muestreo BALANCEADO (200 reseñas por estrella).
La distribución de estrellas no debe extrapolarse a Amazon en general.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import faiss
import numpy as np
import ollama
import pandas as pd
from mcp.server.fastmcp import FastMCP

RAIZ = Path(__file__).resolve().parent
RUTA_CSV = RAIZ / "data" / "resenas.csv"
RUTA_INDICE = RAIZ / "vectorstore" / "resenas.faiss"
RUTA_CONFIG = RAIZ / "vectorstore" / "config.json"

mcp = FastMCP(
    "AmazonReviewsLocal",
    host="127.0.0.1",
    port=8000,
    stateless_http=True,
    json_response=True,
)


@lru_cache(maxsize=1)
def cargar_datos() -> pd.DataFrame:
    """Mantener el pequeño corpus en memoria sin releer el CSV en cada petición."""
    if not RUTA_CSV.is_file():
        raise FileNotFoundError(f"No existe {RUTA_CSV}; ejecuta 01_preparar_resenas.py")
    df = pd.read_csv(RUTA_CSV, dtype={"id": "string"}, encoding="utf-8-sig")
    if df.empty or not {"id", "text", "estrellas"}.issubset(df.columns):
        raise ValueError("CSV vacío o faltan columnas id, text y estrellas")
    if df["id"].duplicated().any() or df["text"].isna().any():
        raise ValueError("Hay IDs duplicados o textos faltantes")
    return df


@lru_cache(maxsize=1)
def cargar_indice_y_config() -> tuple[faiss.Index, dict]:
    """Cargar el índice persistido: la posición i corresponde a la fila i del CSV."""
    if not RUTA_INDICE.is_file() or not RUTA_CONFIG.is_file():
        raise FileNotFoundError("Falta FAISS/config.json; ejecuta 02_construir_indice.py")
    idx = faiss.read_index(str(RUTA_INDICE))
    cfg = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    if idx.ntotal != len(cargar_datos()) or idx.d != int(cfg["dimension"]):
        raise ValueError("Índice, configuración y CSV no coinciden: reconstruye FAISS")
    return idx, cfg


@mcp.tool()
def buscar_resenas(pregunta: str, top_k: int = 5, estrellas: list[int] | None = None) -> dict:
    """Busca reseñas semánticamente similares a una pregunta sobre productos o clientes.

    Usa embeddings locales y el índice FAISS. Devuelve IDs de las reseñas,
    estrellas, textos originales y similitudes. La similitud NO es probabilidad.
    Utiliza los IDs al citar evidencia y no inventes reseñas.

    Args:
        pregunta: consulta en lenguaje natural (español).
        top_k: número de reseñas solicitadas, entre 1 y 8.
        estrellas: filtro opcional obligatorio de calificaciones de 1 a 5; no es un conteo.
    """
    if not pregunta or not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacía")
    if not 1 <= top_k <= 8:
        raise ValueError("top_k debe estar entre 1 y 8")
    if estrellas is not None:
        if not estrellas or any(type(n) is not int or n not in (1, 2, 3, 4, 5) for n in estrellas):
            raise ValueError("estrellas debe contener calificaciones enteras entre 1 y 5")
    df = cargar_datos()
    idx, cfg = cargar_indice_y_config()
    emb = ollama.embed(model=cfg["modelo"], input=pregunta.strip(), keep_alive=0)
    vector = np.ascontiguousarray(np.asarray(emb.embeddings, dtype="float32"))
    if vector.shape != (1, idx.d):
        raise RuntimeError(f"Dimensión inesperada: {vector.shape}; esperada (1, {idx.d})")
    faiss.normalize_L2(vector)
    # El índice incluye todas las calificaciones. Para garantizar el filtro
    # recuperamos el ranking completo y seleccionamos los primeros k válidos.
    # En este corpus de 1.000 reseñas, la búsqueda exhaustiva es pequeña.
    candidatos = idx.ntotal if estrellas is not None else min(top_k, idx.ntotal)
    similitudes, posiciones = idx.search(vector, candidatos)
    fuentes = []
    for score, pos in zip(similitudes[0], posiciones[0]):
        if int(pos) < 0:
            continue
        fila = df.iloc[int(pos)]
        if estrellas is not None and int(fila["estrellas"]) not in estrellas:
            continue
        fuentes.append({
            "id": str(fila["id"]),
            "estrellas": int(fila["estrellas"]),
            "texto_original": str(fila["text"]),
            "similitud_coseno": round(float(score), 4),
        })
        if len(fuentes) >= top_k:
            break
    return {
        "consulta": pregunta,
        "filtro_estrellas": estrellas,
        "fuentes": fuentes,
        "total_resenas_indexadas": int(idx.ntotal),
        "indicacion": "Solo estas reseñas recuperadas constituyen evidencia; cita sus IDs originales.",
    }


@mcp.tool()
def estadisticas_resenas(estrellas: list[int] | None = None, palabra_clave: str = "") -> dict:
    """Calcula estadísticas EXACTAS sobre las 1.000 reseñas locales de Amazon.

    Puede filtrar por una o varias estrellas y una palabra/frase literal (no búsqueda
    semántica). Si estrellas=None, se incluyen las cinco calificaciones.
    Ejemplo: estrellas=[1,2], palabra_clave="entrega".

    IMPORTANTE: la muestra contiene deliberadamente 200 reseñas de cada estrella.
    Sus porcentajes NO representan el porcentaje real de opiniones en Amazon.

    Args:
        estrellas: lista opcional de calificaciones de 1 a 5.
        palabra_clave: texto literal que debe aparecer en la reseña (sin regex).
    """
    df = cargar_datos()
    filtros = df
    if estrellas is not None:
        if not estrellas or any(type(x) is not int or x not in (1, 2, 3, 4, 5) for x in estrellas):
            raise ValueError("estrellas debe ser una lista no vacía de enteros entre 1 y 5")
        filtros = filtros[filtros["estrellas"].isin(estrellas)]
    palabra_clave = palabra_clave.strip()
    if palabra_clave:
        filtros = filtros[
            filtros["text"].str.contains(palabra_clave, regex=False, case=False, na=False)
        ]
    distribucion = {str(e): int((filtros["estrellas"] == e).sum()) for e in range(1, 6)}
    total = len(df)
    return {
        "total_muestra": int(total),
        "reseñas_filtradas": int(len(filtros)),
        "porcentaje_sobre_la_muestra": round(100.0 * len(filtros) / total, 2),
        "promedio_estrellas_filtrado": (round(float(filtros["estrellas"].mean()), 2) if len(filtros) else None),
        "distribucion_estrellas_filtrada": distribucion,
        "filtro_estrellas": estrellas,
        "filtro_palabra_clave": palabra_clave,
        "tipo_busqueda_palabra": "coincidencia literal, no semántica",
        "advertencia_muestreo": (
            "Se eligieron 200 reseñas por estrella (muestreo balanceado). "
            "Los porcentajes calculados solo describen esta muestra; "
            "no son estimaciones de la distribución real de Amazon."
        ),
    }


if __name__ == "__main__":
    # Únicamente accesible desde este computador, no en la red externa.
    print("Servidor MCP Amazon en http://127.0.0.1:8000/mcp", flush=True)
    mcp.run(transport="streamable-http")

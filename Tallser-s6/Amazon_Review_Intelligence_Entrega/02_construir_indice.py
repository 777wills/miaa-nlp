"""Construye y guarda un índice FAISS de 1.000 reseñas usando Ollama local.

Requiere: ejecutar primero 01_preparar_resenas.py y descargar el modelo EmbeddingGemma.
Para recuperar documentos, FAISS devuelve índices alineados con data/resenas.csv.
Uso: python 02_construir_indice.py
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import faiss
import ollama

RAIZ = Path(__file__).resolve().parent
MODELO = "embeddinggemma:300m-qat-q4_0"
BATCH = 16


def main():
    origen = RAIZ / "data" / "resenas.csv"
    if not origen.exists():
        raise FileNotFoundError(f"Primero ejecuta 01_preparar_resenas.py: {origen}")
    df = pd.read_csv(origen, dtype={"id": "string"}, encoding="utf-8-sig")
    if df.empty or df["text"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Corpus vacío, con texto faltante o identificadores repetidos.")
    textos = df["text"].astype(str).tolist()
    matriz = []
    inicio = time.perf_counter()
    print(f"Generando embeddings locales para {len(textos)} reseñas con {MODELO}...")
    for pos in range(0, len(textos), BATCH):
        lote = textos[pos:pos + BATCH]
        respuesta = ollama.embed(model=MODELO, input=lote)
        emb = np.asarray(respuesta.embeddings, dtype="float32")
        if emb.ndim != 2 or emb.shape[0] != len(lote):
            raise RuntimeError(f"Dimensiones inesperadas en lote {pos}: {emb.shape}")
        matriz.append(emb)
        if (pos // BATCH + 1) % 5 == 0 or pos + BATCH >= len(textos):
            print(f"Procesadas {min(pos + BATCH, len(textos))}/{len(textos)}")
    vectores = np.ascontiguousarray(np.vstack(matriz), dtype="float32")
    faiss.normalize_L2(vectores)  # Producto interno de vectores normalizados = coseno
    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(vectores)
    carpeta = RAIZ / "vectorstore"
    carpeta.mkdir(exist_ok=True)
    faiss.write_index(indice, str(carpeta / "resenas.faiss"))
    config = {"modelo": MODELO, "dimension": int(vectores.shape[1]), "filas": int(indice.ntotal),
              "distancia": "coseno (vectores normalizados; IndexFlatIP)",
              "metadata": "data/resenas.csv", "posicion": "fila del CSV (comienza en 0)"}
    (carpeta / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Indice guardado en {carpeta / 'resenas.faiss'}")
    print(f"Vectores: {indice.ntotal} | Dimension: {indice.d} | Tiempo: {time.perf_counter()-inicio:.1f} s")


if __name__ == "__main__":
    main()

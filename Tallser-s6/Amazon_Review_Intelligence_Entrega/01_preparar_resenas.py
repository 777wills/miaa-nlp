"""Prepara un corpus local y reproducible de 1.000 reseñas de Amazon en español.

Fuente: SetFit/amazon_reviews_multi_es (partición oficial train).
Preserva IDs de origen; las etiquetas 0..4 se convierten a estrellas 1..5.
Uso: python 01_preparar_resenas.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datasets import load_dataset

RUTA_BASE = Path(__file__).resolve().parent
RUTA_DATOS = RUTA_BASE / "data"
REPO = "SetFit/amazon_reviews_multi_es"
SPLIT = "train"
SEMILLA = 42
POR_CLASE = 200


def main():
    RUTA_DATOS.mkdir(parents=True, exist_ok=True)
    print(f"Descargando o recuperando del cache: {REPO} [{SPLIT}]...")
    ds = load_dataset(REPO, split=SPLIT)
    columnas = set(ds.column_names)
    requeridas = {"id", "text", "label"}
    if not requeridas.issubset(columnas):
        raise ValueError(f"Faltan columnas: {sorted(requeridas - columnas)}. Encontradas: {ds.column_names}")

    rng = np.random.default_rng(SEMILLA)
    indices = rng.permutation(len(ds))
    seleccionados = []
    totales = {i: 0 for i in range(5)}
    textos_vistos = set()
    ids_vistos = set()
    for indice in indices:
        r = ds[int(indice)]
        etiqueta = int(r["label"])
        if etiqueta not in totales or totales[etiqueta] >= POR_CLASE:
            continue
        texto = str(r["text"] or "").strip()
        id_origen = str(r["id"])
        if not texto or texto in textos_vistos or id_origen in ids_vistos:
            continue
        textos_vistos.add(texto)
        ids_vistos.add(id_origen)
        seleccionados.append({"id": id_origen, "text": texto, "label": etiqueta,
                              "estrellas": etiqueta + 1, "split": SPLIT})
        totales[etiqueta] += 1
        if all(v == POR_CLASE for v in totales.values()):
            break
    if len(seleccionados) != 5 * POR_CLASE:
        raise RuntimeError(f"No fue posible completar la muestra: {totales}")

    df = pd.DataFrame(seleccionados, columns=["id", "text", "label", "estrellas", "split"])
    salida = RUTA_DATOS / "resenas.csv"
    df.to_csv(salida, index=False, encoding="utf-8-sig")
    manifiesto = {"dataset": REPO, "split": SPLIT, "semilla": SEMILLA,
                  "muestra": len(df), "por_estrellas": df["estrellas"].value_counts().sort_index().to_dict(),
                  "archivo": "data/resenas.csv", "ids": "campo id original del dataset"}
    (RUTA_DATOS / "origen.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Archivo guardado: {salida}")
    print(f"Registros: {len(df)} | IDs únicos: {df['id'].nunique()} | Textos únicos: {df['text'].nunique()}")
    print("Distribucion de estrellas:")
    print(df["estrellas"].value_counts().sort_index().to_string())
    print("Primeras reseñas:")
    for _, r in df.head(2).iterrows():
        print(f"- ID={r['id']} | {r['estrellas']} estrellas | {r['text'][:120]}...")


if __name__ == "__main__":
    main()

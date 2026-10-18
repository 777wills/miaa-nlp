"""Descubrimiento local, selección y verificación de las dos Skills del proyecto.

No necesita bibliotecas nuevas ni crea herramientas MCP.
Los contenidos SKILL.md no se cargan solos en Ollama: el orquestador importa este módulo.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import csv
import math

RAIZ = Path(__file__).resolve().parent
DIRECTORIO = RAIZ / "skills"
AUTORIZADAS = ("analisis-voz-consumidor", "auditoria-evidencia-rag")


@dataclass(frozen=True)
class Skill:
    nombre: str
    descripcion: str
    instrucciones: str
    ruta: Path


def _leer_skill(nombre: str) -> Skill:
    if nombre not in AUTORIZADAS:
        raise ValueError(f"Skill no reconocida: {nombre}")
    ruta = DIRECTORIO / nombre / "SKILL.md"
    contenido = ruta.read_text(encoding="utf-8")
    if len(contenido) > 12_000:
        raise ValueError(f"Skill demasiado grande: {nombre}")
    partes = contenido.split("---", 2)
    if len(partes) != 3 or partes[0].strip():
        raise ValueError(f"La Skill {nombre} requiere encabezado ---")
    campos: dict[str, str] = {}
    for linea in partes[1].splitlines():
        if ":" in linea:
            clave, valor = linea.split(":", 1)
            campos[clave.strip()] = valor.strip()
    if campos.get("name") != nombre or not campos.get("description"):
        raise ValueError(f"Encabezado SKILL.md incorrecto: {nombre}")
    instrucciones = partes[2].strip()
    if not instrucciones:
        raise ValueError(f"Skill vacía: {nombre}")
    return Skill(nombre, campos["description"], instrucciones, ruta)


@lru_cache(maxsize=1)
def cargar_skills() -> dict[str, Skill]:
    """Lee solamente las dos rutas explícitamente autorizadas."""
    return {nombre: _leer_skill(nombre) for nombre in AUTORIZADAS}


def seleccionar_skills(*, necesita_fuentes: bool, necesita_estadisticas: bool) -> list[str]:
    """Reglas de activación transparentes y reproducibles."""
    seleccion = []
    if necesita_fuentes:
        seleccion.append("analisis-voz-consumidor")
    if necesita_fuentes or necesita_estadisticas:
        seleccion.append("auditoria-evidencia-rag")
    return seleccion


def instrucciones_activas(nombres: list[str]) -> str:
    catalogo = cargar_skills()
    partes = []
    for nombre in nombres:
        if nombre not in catalogo:
            raise ValueError(f"Skill no autorizada: {nombre}")
        partes.append(f"SKILL: {nombre}\n{catalogo[nombre].instrucciones}")
    return "\n\n".join(partes)


@lru_cache(maxsize=1)
def _corpus_por_id() -> dict[str, dict[str, str]]:
    ruta = RAIZ / "data" / "resenas.csv"
    with ruta.open("r", encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    if len({fila["id"] for fila in filas}) != len(filas):
        raise ValueError("El corpus contiene IDs duplicados")
    return {fila["id"]: fila for fila in filas}


def auditar_resultados(resultados: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Revisa evidencia y aritmética antes de mostrarlas.

    Deja los cálculos en MCP, verifica que lo recibido es coherente y rastreable.
    """
    corpus = _corpus_por_id()
    comprobaciones: list[str] = []
    for nombre, datos in resultados:
        if nombre == "buscar_resenas":
            fuentes = datos.get("fuentes", [])
            if not isinstance(fuentes, list):
                raise ValueError("MCP devolvió fuentes con formato inesperado")
            vistos = set()
            filtro = datos.get("filtro_estrellas")
            if filtro is not None and not isinstance(filtro, list):
                raise ValueError("Filtro de estrellas inválido en resultado MCP")
            for fuente in fuentes:
                id_ = str(fuente.get("id", ""))
                fila = corpus.get(id_)
                if fila is None or id_ in vistos:
                    raise ValueError(f"Fuente inexistente o duplicada: {id_}")
                vistos.add(id_)
                if str(fuente.get("texto_original")) != fila["text"]:
                    raise ValueError(f"El texto no coincide con el corpus: {id_}")
                if int(fuente.get("estrellas")) != int(fila["estrellas"]):
                    raise ValueError(f"La calificación no coincide con el corpus: {id_}")
                if filtro is not None and int(fuente["estrellas"]) not in filtro:
                    raise ValueError(f"La reseña {id_} no cumple el filtro de estrellas")
            comprobaciones.append(f"{len(fuentes)} fuentes contrastadas con sus IDs originales")
        elif nombre == "estadisticas_resenas":
            total = int(datos["total_muestra"])
            n = int(datos["reseñas_filtradas"])
            porcentaje = float(datos["porcentaje_sobre_la_muestra"])
            if total != len(corpus) or not (0 <= n <= total):
                raise ValueError("El total o el conteo no coinciden con el corpus")
            if not math.isclose(porcentaje, round(100.0 * n / total, 2), abs_tol=0.011):
                raise ValueError("Porcentaje recibido incoherente con conteo y total")
            estrellas = datos.get("filtro_estrellas")
            palabra = str(datos.get("filtro_palabra_clave") or "")
            if estrellas is not None and (not isinstance(estrellas, list) or
                    any(type(v) is not int or v not in (1, 2, 3, 4, 5) for v in estrellas)):
                raise ValueError("Filtro estadístico de estrellas inválido")
            # Validación independiente del recuento original de MCP (también
            # cuando haya una palabra literal). Evita validar únicamente que
            # el porcentaje sea consistente con un conteo erróneo.
            filas = [fila for fila in corpus.values()
                     if (estrellas is None or int(fila["estrellas"]) in estrellas)
                     and (not palabra or palabra.casefold() in fila["text"].casefold())]
            if n != len(filas):
                raise ValueError("El conteo literal no coincide con las reseñas del corpus")
            distribucion = datos.get("distribucion_estrellas_filtrada")
            if distribucion is not None:
                esperado_dist = {str(e): sum(int(f["estrellas"]) == e for f in filas)
                                 for e in range(1, 6)}
                if {str(k): int(v) for k, v in distribucion.items()} != esperado_dist:
                    raise ValueError("La distribución por estrellas no coincide")
            comprobaciones.append("Conteo, filtros literales y porcentajes comprobados")
        else:
            raise ValueError(f"Herramienta inesperada durante auditoría: {nombre}")
    return comprobaciones


if __name__ == "__main__":
    for k, s in cargar_skills().items():
        print(f"{k}: {s.descripcion} ({s.ruta})")
    print("Selección cualitativa:", seleccionar_skills(necesita_fuentes=True, necesita_estadisticas=False))
    print("Selección estadística:", seleccionar_skills(necesita_fuentes=False, necesita_estadisticas=True))

"""Pruebas locales del gestor: no requieren Ollama, MCP ni Internet.

Para probar Gradio y las consultas reales, ejecutar la aplicación V4 en Windows.
"""
from __future__ import annotations
import csv
import importlib
import tempfile
from pathlib import Path

m = importlib.import_module("08_gestor_skills")


def comprobar():
    catalogo = m.cargar_skills()
    assert set(catalogo) == {"analisis-voz-consumidor", "auditoria-evidencia-rag"}
    assert all(catalogo[n].instrucciones.strip() for n in catalogo)
    assert m.seleccionar_skills(necesita_fuentes=True, necesita_estadisticas=False) == [
        "analisis-voz-consumidor", "auditoria-evidencia-rag"]
    assert m.seleccionar_skills(necesita_fuentes=False, necesita_estadisticas=True) == [
        "auditoria-evidencia-rag"]
    assert m.seleccionar_skills(necesita_fuentes=False, necesita_estadisticas=False) == []
    assert "ID" in m.instrucciones_activas(["auditoria-evidencia-rag"])

    # Corpus mínimo sintético para probar reglas sin tocar las reseñas reales.
    with tempfile.TemporaryDirectory() as t:
        carpeta = Path(t)
        (carpeta / "data").mkdir()
        with (carpeta / "data" / "resenas.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "text", "estrellas"])
            w.writeheader()
            w.writerow({"id": "es_0001", "text": "Buena mochila", "estrellas": 4})
            w.writerow({"id": "es_0002", "text": "Muy mala entrega", "estrellas": 1})
        ruta_original = m.RAIZ
        try:
            m.RAIZ = carpeta
            m._corpus_por_id.cache_clear()
            busqueda = {"fuentes": [{"id": "es_0001", "estrellas": 4,
                                        "texto_original": "Buena mochila"}],
                        "filtro_estrellas": [4]}
            stats = {"total_muestra": 2, "reseñas_filtradas": 1,
                     "porcentaje_sobre_la_muestra": 50,
                     "filtro_estrellas": [4], "filtro_palabra_clave": ""}
            correctos = m.auditar_resultados([
                ("buscar_resenas", busqueda), ("estadisticas_resenas", stats)])
            assert len(correctos) == 2
            for fuente_invalida in [
                {"id": "es_9999", "estrellas": 4, "texto_original": "Buena mochila"},
                {"id": "es_0001", "estrellas": 5, "texto_original": "Buena mochila"},
                {"id": "es_0001", "estrellas": 4, "texto_original": "Otro texto"},
            ]:
                try:
                    m.auditar_resultados([("buscar_resenas", {"fuentes": [fuente_invalida],
                                                             "filtro_estrellas": [4]})])
                except ValueError:
                    pass
                else:
                    raise AssertionError("La auditoría aceptó evidencia alterada")
            try:
                m.auditar_resultados([("estadisticas_resenas", {**stats, "porcentaje_sobre_la_muestra": 80})])
            except ValueError:
                pass
            else:
                raise AssertionError("La auditoría aceptó un porcentaje inconsistente")
        finally:
            m.RAIZ = ruta_original
            m._corpus_por_id.cache_clear()

    print("OK: 2 Skills, selección por intención, carga de instrucciones, auditoría de fuentes y porcentajes")


if __name__ == "__main__":
    comprobar()

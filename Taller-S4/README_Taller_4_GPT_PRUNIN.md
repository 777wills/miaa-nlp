# Taller 4 — Generación de texto con GPT aplicada a PRUNIN

Este proyecto corresponde al **Taller 4 del curso Deep Learning Avanzado** y busca poner en práctica el funcionamiento de los modelos tipo GPT para generación de texto.

La idea del ejercicio fue trabajar con un caso diferente a los ejemplos de clase y llevarlo a un contexto cercano a PRUNIN. Para esto se utilizaron **issues reales de GitHub** como fuente de entrenamiento, con el objetivo de observar si un modelo GPT pequeño podía aprender parte de la estructura y del vocabulario usado en incidencias técnicas.

## Objetivo

El propósito principal fue construir y evaluar un modelo generativo capaz de continuar textos relacionados con incidencias de proyectos de software.

La pregunta que orientó el trabajo fue:

> ¿Puede un modelo tipo GPT aprender los patrones lingüísticos presentes en incidencias reales de proyectos de software y generar nuevos textos coherentes con ese dominio?

También se comparó cómo cambia el comportamiento del modelo al modificar su profundidad y algunos parámetros de generación.

## Dataset

Se utilizó el dataset público de Hugging Face:

`noamaanMulla-03/datasets-issues`

El conjunto contiene información real obtenida desde GitHub. Para este trabajo se filtraron únicamente los registros correspondientes a **issues**, excluyendo los pull requests.

Después de eliminar títulos vacíos y duplicados, el corpus quedó con:

- 3.276 issues
- 2.620 registros para entrenamiento
- 327 para validación
- 329 para test

Los textos se organizaron con una estructura sencilla:

```text
<TITLE> título del issue
<BODY> descripción del issue
```

## Exploración de los datos

Antes del entrenamiento se realizó un análisis exploratorio para entender mejor el corpus.

Se revisaron, entre otros aspectos:

- cantidad de issues disponibles;
- presencia o ausencia de descripción;
- longitud de títulos y cuerpos;
- palabras más frecuentes;
- bigramas frecuentes;
- labels presentes en el dataset.

Este análisis permitió confirmar que el corpus está fuertemente orientado a un contexto técnico, con términos relacionados con datasets, errores, carga de información y solicitudes de nuevas funcionalidades.

## Modelo

Se utilizó una arquitectura basada en GPT implementada con `GPT2LMHeadModel`.

El tokenizador parte de `GPT2TokenizerFast`, pero los pesos del modelo generativo se inicializan desde cero. De esta forma, el entrenamiento realizado en el notebook corresponde al corpus seleccionado para el taller.

Se probaron dos configuraciones:

| Configuración | Capas | Heads | Embedding | Dropout | Learning rate | Épocas |
|---|---:|---:|---:|---:|---:|---:|
| Mini-GPT 2 capas | 2 | 4 | 128 | 0.1 | 0.0003 | 3 |
| Mini-GPT 4 capas | 4 | 4 | 128 | 0.1 | 0.0003 | 3 |

## Resultados

Al comparar los dos modelos, el de 4 capas obtuvo un resultado ligeramente mejor, aunque la diferencia fue pequeña.

| Modelo | Validation Loss | Validation Perplexity |
|---|---:|---:|
| Mini-GPT 4 capas | 4.8012 | 121.65 |
| Mini-GPT 2 capas | 4.8099 | 122.71 |

El modelo seleccionado fue el **Mini-GPT de 4 capas**.

En el conjunto de test obtuvo:

- Test Loss: **4.7671**
- Test Perplexity: **117.57**

En esta corrida, aumentar de 2 a 4 capas produjo una mejora pequeña. Por eso, los resultados no indican que una mayor profundidad sea necesariamente mejor en todos los casos.

## Generación de texto

Además del entrenamiento, se probaron diferentes formas de generación:

- greedy decoding;
- sampling;
- temperature;
- top-k.

Al revisar las salidas, el modelo sí aprende parte del vocabulario y de la estructura de los issues. Por ejemplo, aparecen expresiones como:

- `dataset`
- `load_dataset`
- `Describe the bug`
- `Steps to reproduce the bug`

Sin embargo, también se presentan repeticiones, frases incompletas y construcciones poco naturales. Esto muestra que el modelo logró aprender patrones del dominio, pero todavía tiene limitaciones importantes de coherencia.

También se calcularon indicadores de diversidad léxica como **Distinct-1** y **Distinct-2** para comparar las generaciones. Los resultados mostraron que aumentar la diversidad no necesariamente produce textos más coherentes.

## Prueba manual

El notebook incluye una función interactiva que permite escribir el inicio de una incidencia y observar cómo el modelo continúa el texto.

Por ejemplo:

```python
demo_interactiva()
```

En una de las pruebas se ingresó:

```text
La aplicación falla cuando
```

El modelo procesó el texto, pero continuó principalmente en inglés. Esto es esperable porque casi todo el corpus de entrenamiento está en ese idioma.

Esta prueba sirve también para mostrar una limitación del modelo: puede recibir un prompt en español, pero no fue entrenado para mantener una generación bilingüe consistente.

## Relación con PRUNIN

El ejercicio no busca convertir este modelo en una solución de producción.

La utilidad para PRUNIN está en probar un flujo completo de generación de texto a partir de información real:

```text
issues reales
      ↓
preprocesamiento
      ↓
tokenización
      ↓
entrenamiento GPT
      ↓
evaluación
      ↓
generación de nuevos textos
```

A futuro, un enfoque similar podría utilizarse para generar borradores de incidencias, descripciones iniciales o escenarios técnicos que posteriormente sean revisados por una persona.

## Conclusiones

El experimento permitió comprobar que un GPT pequeño puede aprender parte de los patrones presentes en un corpus real de incidencias técnicas.

El modelo de 4 capas obtuvo resultados ligeramente mejores que el de 2 capas, aunque la diferencia fue reducida. Las generaciones muestran que el modelo reconoce vocabulario y estructuras del dominio, pero todavía produce repeticiones y frases poco naturales.

La principal conclusión es que el modelo funciona como una demostración del proceso de generación autoregresiva aprendido en clase, pero no debe considerarse una herramienta lista para producción.

## Ejecución

Para ejecutar el notebook desde cero:

1. Abrir `Taller_4_GPT_PRUNIN_GitHub_Issues(3).ipynb`.
2. Seleccionar un kernel de Python.
3. Ejecutar **Restart Kernel and Run All**.
4. Esperar a que finalice el entrenamiento.
5. Revisar los resultados y las generaciones.
6. Ejecutar `demo_interactiva()` de forma manual si se desea probar un prompt propio.

El notebook descarga el dataset directamente desde Hugging Face, por lo que requiere conexión a Internet durante la carga inicial.

## Tecnologías utilizadas

- Python
- PyTorch
- Hugging Face Datasets
- Transformers
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

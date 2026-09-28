# Taller 4 — Generación de texto con GPT aplicada a issues de GitHub

Este proyecto corresponde al **Taller 4 del curso de Procesamiento de Lenguaje Natural** de la Maestría en Inteligencia Artificial Aplicada y busca poner en práctica el funcionamiento de los modelos tipo GPT para generación de texto.

El trabajo utiliza **issues reales de GitHub** como fuente de entrenamiento, con el objetivo de observar si un modelo GPT pequeño puede aprender parte de la estructura y del vocabulario usado en incidencias técnicas.

## Objetivo

El propósito principal fue construir y evaluar un modelo generativo capaz de continuar textos relacionados con incidencias de proyectos de software.

La pregunta que orientó el trabajo fue:

> ¿Puede un modelo tipo GPT aprender los patrones lingüísticos presentes en incidencias reales de proyectos de software y generar nuevos textos coherentes con ese dominio?

También se comparó cómo cambia el comportamiento del modelo al modificar su profundidad y algunos parámetros de generación.

## Dataset

Se utilizó el dataset público de Hugging Face:

`noamaanMulla-03/datasets-issues`

El conjunto contiene 8.084 registros obtenidos con la API REST de GitHub a partir del repositorio `huggingface/datasets`, entre issues y pull requests. Para este trabajo se filtraron únicamente los registros correspondientes a **issues**, excluyendo los pull requests, lo que deja 3.293 incidencias.

Después de eliminar títulos vacíos y 17 títulos duplicados, el corpus final contiene 3.276 issues: 2.620 para entrenamiento, 327 para validación y 329 para test.

Los textos se organizaron con una estructura sencilla:

```text
<TITLE> título del issue
<BODY> descripción del issue
<EOS>
```

`<TITLE>` y `<BODY>` se registran como tokens especiales. Cada documento termina con EOS y se utiliza un token PAD separado.

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

- Solo 22 issues no tienen descripción.
- Los títulos son cortos (mediana de 7 palabras), mientras que las descripciones tienen una mediana de 128 palabras y una cola larga que llega a más de 12.000.
- Los labels más frecuentes son `bug` (706), `enhancement` (501) y `dataset request` (161).
- Una vez tokenizado, cada documento tiene una mediana de 155 tokens y un percentil 90 de 329 tokens.

## Modelo

Se utilizó una arquitectura basada en GPT implementada con `GPT2LMHeadModel`.

El tokenizador parte de `GPT2TokenizerFast`, pero los pesos del modelo generativo se inicializan desde cero. De esta forma, el entrenamiento realizado en el notebook corresponde al corpus seleccionado para el taller.

Se probaron dos configuraciones, con un máximo de diez épocas y early stopping con paciencia de dos épocas:

| Configuración    | Capas | Heads | Embedding | Dropout | Learning rate |    Épocas |
| ---------------- | ----: | ----: | --------: | ------: | ------------: | --------: |
| Mini-GPT 2 capas |     2 |     4 |       128 |     0.1 |        0.0003 | 10 máximo |
| Mini-GPT 4 capas |     4 |     4 |       128 |     0.1 |        0.0003 | 10 máximo |

El modelo de 2 capas tiene 6.862.848 parámetros y el de 4 capas 7.259.392. La mayor parte corresponde a la matriz de embeddings de tokens (50.260 × 128 ≈ 6,43 millones, compartida con la capa de salida), por lo que duplicar la profundidad solo añade unos 400.000 parámetros.

## Resultados

La distribución tokenizada llevó a utilizar `BLOCK_SIZE = 256`, con una cobertura de 76.53%; ningún candidato entre 128, 192 y 256 alcanzó el 90%, por lo que se eligió el mayor bloque disponible.

| Modelo           | Épocas | Validation loss | Validation perplexity |   Tiempo |
| ---------------- | -----: | --------------: | --------------------: | -------: |
| Mini-GPT 2 capas |     10 |          4.2441 |                 69.69 | 197.29 s |
| Mini-GPT 4 capas |     10 |          4.2297 |                 68.69 | 232.98 s |

Ambos modelos llegaron a la época 10 sin activar early stopping y su validation loss seguía bajando en la última época, lo que indica que todavía tenían margen de mejora con más entrenamiento. Se seleccionó el Mini-GPT de 4 capas, que obtuvo test loss 4.0919 y test perplexity 59.85. Los tiempos corresponden a un equipo con aceleración MPS y dependen del hardware.

## Generación de texto

Además del entrenamiento, se probaron diferentes formas de generación:

- greedy decoding;
- temperatura;
- top-k;
- top-p o nucleus sampling;
- temperatura + top-k.

Al revisar las salidas, el modelo sí aprende parte del vocabulario y de la estructura de los issues. Por ejemplo, aparecen expresiones como:

- `dataset`
- `load_dataset`
- `Describe the bug`
- `Feature request`
- `Adding a Dataset`

Sin embargo, también se presentan repeticiones, frases incompletas y tokens inexistentes como `load_andaset`, que combinan fragmentos subword de palabras frecuentes.

También se calcularon indicadores de diversidad léxica como **Distinct-1** y **Distinct-2** para comparar las generaciones:

| Estrategia                     | Distinct-1 | Distinct-2 |
| ------------------------------ | ---------: | ---------: |
| Greedy                         |      0.481 |      0.577 |
| Temperatura (0.7)              |      0.586 |      0.750 |
| Top-k (40)                     |      0.852 |      1.000 |
| Top-p (0.9)                    |      0.816 |      1.000 |
| Temperatura (0.8) + top-k (40) |      0.676 |      0.917 |

Greedy entra en un bucle que repite la plantilla `- **Description:** *link to the dataset*`, mientras que top-k y top-p producen textos más variados pero menos coherentes. Aumentar la diversidad no necesariamente produce textos más coherentes; la combinación de temperatura y top-k fue el punto intermedio que se utilizó para el resto de las generaciones.

Ninguna generación terminó en EOS. Con 45 a 50 tokens nuevos el modelo no alcanza a completar un issue, cuya mediana es de 155 tokens.

## Demo manual

El notebook incluye una función interactiva para escribir el inicio de una incidencia y observar cómo el modelo continúa el texto. Se ejecuta manualmente al finalizar:

```python
demo_interactiva()
```

## Flujo del experimento

El ejercicio prueba un flujo completo de generación de texto a partir de información real:

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

Un enfoque similar podría utilizarse para generar borradores de incidencias o descripciones iniciales que posteriormente sean revisados por una persona.

## Conclusiones

El experimento permitió comprobar que un GPT pequeño puede aprender parte de los patrones presentes en un corpus real de incidencias técnicas.

El modelo de 4 capas obtuvo resultados ligeramente mejores que el de 2 capas, aunque la diferencia fue reducida (0.014 de validation loss). Como casi todos los parámetros están en los embeddings, la profundidad cambia poco la capacidad total del modelo, y ninguno de los dos llegó a converger en diez épocas.

Las generaciones muestran que el modelo reconoce vocabulario y estructuras del dominio, como las plantillas de reporte de errores y de solicitud de datasets, pero todavía produce repeticiones y frases poco naturales.

La principal conclusión es que el modelo funciona como una demostración del proceso de generación autoregresiva, pero no debe considerarse una herramienta lista para producción. El experimento está limitado por el tamaño del corpus, el uso de un único repositorio y la ausencia de una evaluación humana de la corrección técnica de los textos.

## Ejecución

Para ejecutar el notebook desde cero:

1. Abrir `Taller_4_GPT_GitHub_Issues.ipynb`.
2. Seleccionar un kernel de Python.
3. Ejecutar **Restart Kernel and Run All**.
4. Esperar a que finalice el entrenamiento.
5. Revisar los resultados y las generaciones.
6. Ejecutar `demo_interactiva()` de forma manual si se desea probar un prompt propio.

El notebook descarga el dataset directamente desde Hugging Face, por lo que requiere conexión a Internet durante la carga inicial. Al terminar guarda el modelo seleccionado y el tokenizador en `artifacts_taller4_gpt_github_issues/`.

Los resultados reportados se obtuvieron con aceleración MPS. El notebook también funciona en CPU y CUDA; con la misma semilla los resultados deberían ser cercanos, aunque pueden existir pequeñas variaciones entre dispositivos.

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

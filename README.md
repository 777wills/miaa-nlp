# Procesamiento de Lenguaje Natural — MIAA

Talleres del curso de NLP de la Maestría en Inteligencia Artificial Aplicada. Cada taller
semanal vive en su propia carpeta `Taller-sN/`, con el cuaderno y los recursos que le
corresponden.

## Estudiantes

- William Alberto Suaza Losada
- Anderson Trujillo
- Bairon Gutiérrez

## Estructura del repositorio

```
miaa-nlp/
├── data/              # archivos de apoyo pequeños (los corpus grandes no se versionan)
│   ├── ley_769_2002_articulos.jsonl        # Código de Tránsito limpio (semana 5)
│   └── preguntas_evaluacion_transito.csv   # set de evaluación del RAG (semana 5)
├── requirements.txt   # dependencias comunes a todos los talleres
├── Taller-s1/         # semana 1
│   └── clasificacion_resenas_es_lstm_gru.ipynb
├── Taller-s2/         # semana 2
│   └── clasificacion_resenas_es_transformers.ipynb
├── Taller-s3/         # semana 3
│   └── Proyecto_bert_resenas_amazon_colab_Experimentos.ipynb
├── Taller-S4/         # semana 4
│   ├── Taller_4_GPT_GitHub_Issues.ipynb
│   └── README_Taller_4_GPT_GitHub_Issues.md
├── Taller-s5/         # semana 5
│   └── asistente_codigo_transito_rag.ipynb
├── Tallser-s6/        # semana 6
│   ├── 01_Amazon_Review_Intelligence.ipynb
│   └── Amazon_Review_Intelligence_Entrega/
│       ├── 01_preparar_resenas.py        # muestra balanceada de 1.000 reseñas
│       ├── 02_construir_indice.py        # embeddings + índice FAISS
│       ├── 04_servidor_mcp_v3_2.py       # servidor MCP con dos herramientas
│       ├── 07_app_gradio_v4_skills.py    # chat Gradio + Qwen3 + Skills
│       ├── 08_gestor_skills.py           # carga, selección y auditoría de Skills
│       ├── 09_probar_skills.py           # pruebas del gestor de Skills
│       ├── requirements.txt
│       └── skills/
│           ├── analisis-voz-consumidor/SKILL.md
│           └── auditoria-evidencia-rag/SKILL.md
└── README.md
```

Cada semana nueva añade una carpeta `Taller-sN/` siguiendo el mismo patrón.

## Talleres

### Taller-s1 — Predicción de la calificación de reseñas en español con redes recurrentes

Clasificación ordinal de reseñas de productos en español (1 a 5 estrellas) comparando tres
niveles de modelado: bolsa de palabras, redes recurrentes con embeddings aprendidos desde
cero y redes recurrentes con vectores preentrenados.

- **Cuaderno:** [Taller-s1/clasificacion_resenas_es_lstm_gru.ipynb](Taller-s1/clasificacion_resenas_es_lstm_gru.ipynb),
  con todo el desarrollo, desde la auditoría del corpus hasta la demo interactiva.
- **Corpus:** [`SetFit/amazon_reviews_multi_es`](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es)
  del Hub de Hugging Face: 200.000 reseñas de entrenamiento y 5.000 de validación y de
  prueba, con la calificación codificada de 0 a 4. Es la versión en español del corpus
  `amazon_reviews_multi`, reducida a las columnas relevantes.
- **Resultado:** el modelo seleccionado es una GRU con embeddings aprendidos desde cero.
  Sobre la partición de prueba obtiene 0,5066 de F1-macro, 0,6037 de MAE en estrellas y
  0,7798 de kappa cuadrático ponderado.
- **Salidas:** al ejecutarlo se generan `csv_logs/` y `tb_logs/` con las métricas de
  entrenamiento, y `artefactos_resenas/` con los pesos del modelo final y sus metadatos.

### Taller-s2 — Clasificación de reseñas en español con un encoder Transformer

Mismo problema ordinal que la semana 1, abordado implementando desde cero el codificador
del Transformer —codificación posicional sinusoidal, atención multi-cabeza y bloque con
conexiones residuales— y comparándolo contra modelos sin atención.

- **Cuaderno:** [Taller-s2/clasificacion_resenas_es_transformers.ipynb](Taller-s2/clasificacion_resenas_es_transformers.ipynb),
  con diecisiete entrenamientos, ablaciones de las piezas del modelo, curva de datos,
  análisis de los mapas de atención y demo interactiva.
- **Corpus:** el mismo `SetFit/amazon_reviews_multi_es`, con un tokenizador subword BPE de
  16.000 entradas entrenado sobre las propias reseñas y corte de secuencia en 86 tokens.
- **Comparación:** modelo lineal sobre TF-IDF, dos perceptrones multicapa, una GRU y nueve
  variantes del Transformer, más dos programaciones alternativas del ritmo de aprendizaje y
  una versión entrenada con una pérdida sensible al orden de las clases. Todos comparten
  optimizador, ritmo de aprendizaje, recorte de gradiente, criterio de parada, semilla y
  orden de los lotes.
- **Aporte de la atención:** en lugar de comparar el Transformer contra un modelo sin
  bloques, que no aísla la atención porque al quitar el bloque desaparecen todas sus piezas
  a la vez, el cuaderno monta una escalera de cuatro controles con el mismo número de
  parámetros. Mezclar entre posiciones recupera lo que pierde un bloque que transforma cada
  posición por separado, y que esa mezcla se **aprenda** del contenido añade **+0,0120 de
  kappa** (0,7456 frente a 0,7336 con pesos uniformes), una diferencia que queda por debajo
  del margen de error.
- **Resultado:** con 50.000 reseñas ningún Transformer supera a la bolsa de palabras —0,7503
  de kappa frente a 0,7805 del perceptrón sobre TF-IDF y 0,7509 de la regresión logística,
  que se entrena en menos de diez segundos de CPU—, pero al cuadruplicar el corpus el
  Transformer llega a 0,7803 y la alcanza. El modelo seleccionado es la GRU entrenada con el
  corpus completo, que sobre la partición de prueba obtiene 0,5347 de F1-macro, 0,5643 de
  MAE en estrellas y 0,7986 de kappa cuadrático ponderado, con el 92,0% de las predicciones
  a menos de una estrella del valor real. El Transformer entrenado con el mismo corpus queda
  por delante en prueba en las cuatro métricas (0,8008 de kappa), aunque todas las
  diferencias son menores que el margen de error.
- **Salidas:** `csv_logs/` y `tb_logs/` con las métricas de entrenamiento, y
  `artefactos_transformer/` con los pesos del modelo final, el tokenizador y sus metadatos.

### Taller-s3 — Clasificación de reseñas en español con variantes de BERT

Mismo problema de 1 a 5 estrellas, abordado ahora con _fine-tuning_ completo de modelos BERT
preentrenados. El experimento es controlado: se mantienen fijos los datos, las particiones y
los hiperparámetros, y solo cambia el checkpoint.

- **Cuaderno:** [Taller-s3/Proyecto_bert_resenas_amazon_colab_Experimentos.ipynb](Taller-s3/Proyecto_bert_resenas_amazon_colab_Experimentos.ipynb),
  con la depuración del corpus, un EDA breve, el análisis de longitud en tokens, los tres
  experimentos y la comparación de métricas y patrones de error.
- **Corpus:** el mismo `SetFit/amazon_reviews_multi_es`. Se eliminan textos vacíos y
  duplicados, se retiran del entrenamiento las reseñas que aparecen en validación o prueba
  y se toma una muestra balanceada de 50.000 reseñas (10.000 por clase). Validación y prueba
  conservan las particiones originales.
- **Modelos:** BETO Cased (`dccuchile/bert-base-spanish-wwm-cased`), BETO Uncased
  (`dccuchile/bert-base-spanish-wwm-uncased`) y Multilingual BERT
  (`bert-base-multilingual-cased`), todos con `Trainer` de Hugging Face, 128 tokens como
  máximo, 2 épocas, tasa de aprendizaje de `2e-5`, _weight decay_ de 0,01, lotes de 8 y
  selección del mejor checkpoint por _accuracy_ en validación. Con ese corte se trunca menos
  del 3% de las reseñas en los tres tokenizadores.
- **Resultado sobre la partición de prueba:**

  | Modelo            | Accuracy | MAE (estrellas) | Tiempo de entrenamiento |
  | ----------------- | -------- | --------------- | ----------------------- |
  | BETO Cased        | 0,5829   | 0,4794          | 13,73 min               |
  | BETO Uncased      | 0,5757   | 0,4874          | 13,92 min               |
  | Multilingual BERT | 0,5667   | 0,5102          | 17,13 min               |

  BETO Cased es el mejor en las tres medidas y deja el 94,9% de las predicciones a una
  estrella o menos del valor real. La ventaja sobre BETO Uncased es pequeña; el modelo
  multilingüe queda por detrás y es el más lento, porque su tokenizador parte las reseñas en
  más piezas (42 tokens de media frente a 37).

- **Salidas:** `bert_amazon_experimento/` (en `/content/` si se ejecuta en Colab) con las
  métricas en JSON, las predicciones sobre prueba en CSV y la matriz de confusión de cada
  modelo. Los checkpoints intermedios se borran al terminar cada entrenamiento.

### Taller-S4 — Generación de incidencias de software con un GPT entrenado desde cero

Modelo generativo tipo GPT que aprende a redactar issues de GitHub a partir de un título
incompleto. A diferencia de las semanas anteriores, la tarea es generación autoregresiva:
el modelo predice el siguiente token usando solo los anteriores.

- **Cuaderno:** [Taller-S4/Taller_4_GPT_GitHub_Issues.ipynb](Taller-S4/Taller_4_GPT_GitHub_Issues.ipynb),
  con la selección de issues, el EDA, la elección de la longitud de contexto, el
  entrenamiento de dos configuraciones, la comparación de estrategias de decodificación y
  una demo interactiva. El detalle está en
  [Taller-S4/README_Taller_4_GPT_GitHub_Issues.md](Taller-S4/README_Taller_4_GPT_GitHub_Issues.md).
- **Corpus:** [`noamaanMulla-03/datasets-issues`](https://huggingface.co/datasets/noamaanMulla-03/datasets-issues),
  issues y pull requests del repositorio `huggingface/datasets`. Se conservan solo los issues
  con título único: 3.276 documentos con el formato `<TITLE> título <BODY> descripción <eos>`,
  divididos en 2.620, 327 y 329 para entrenamiento, validación y prueba.
- **Modelos:** `GPT2LMHeadModel` inicializado desde cero sobre el tokenizador de GPT-2,
  ampliado con `<TITLE>`, `<BODY>` y un token de relleno propio. Se comparan versiones de 2 y
  4 capas (4 cabezas, embeddings de 128, contexto de 256 tokens), con un máximo de 10 épocas
  y parada temprana con paciencia de 2.
- **Resultado:** el modelo de 4 capas queda apenas por delante (68,69 de perplejidad en
  validación frente a 69,69) y sobre la partición de prueba obtiene 4,0919 de pérdida y
  59,85 de perplejidad. Ninguno activó la parada temprana. Las generaciones reproducen
  vocabulario y plantillas del repositorio (`Describe the bug`, `Adding a Dataset`,
  `load_dataset`), pero con repeticiones y palabras inventadas.

  | Decodificación      | Distinct-1 | Distinct-2 |
  | ------------------- | ---------- | ---------- |
  | Greedy              | 0,481      | 0,577      |
  | Temperatura 0,7     | 0,586      | 0,750      |
  | Top-k 40            | 0,852      | 1,000      |
  | Top-p 0,9           | 0,816      | 1,000      |
  | Temperatura + top-k | 0,676      | 0,917      |

- **Salidas:** `artifacts_taller4_gpt_github_issues/` con los pesos del modelo seleccionado y
  el tokenizador.

### Taller-s5 — Asistente conversacional sobre el Código Nacional de Tránsito con RAG

Chatbot que responde preguntas coloquiales sobre la Ley 769 de 2002 citando los artículos
en los que se apoya. Combina recuperación con FAISS, re-ranking con un cross-encoder y un
modelo generador servido localmente con Ollama, orquestados con LangChain y expuestos en
una interfaz de Gradio.

- **Cuaderno:** [Taller-s5/asistente_codigo_transito_rag.ipynb](Taller-s5/asistente_codigo_transito_rag.ipynb),
  con la limpieza del corpus, el análisis exploratorio, los experimentos de recuperación, la
  comparación de generadores, la calibración del umbral de abstención, la conversación con
  memoria y el chatbot.
- **Corpus:** texto consolidado de la ley en el Gestor Normativo de Función Pública, limpio
  de versiones anteriores y notas de vigencia y jurisprudencia: 180 artículos, 176 vigentes y
  28.501 palabras. Se guarda como snapshot en `data/ley_769_2002_articulos.jsonl`; la
  descarga en vivo queda como respaldo.
- **Evaluación:** 46 preguntas del código con su artículo correcto y una frase de evidencia,
  más 6 preguntas externas (`data/preguntas_evaluacion_transito.csv`). Además de Recall@k y
  MRR@10, se mide la **Evidencia@k**: si entre los k fragmentos hay uno que contiene la frase
  que responde la pregunta.
- **Comparación:** tres fragmentaciones (artículo completo, ventana de 100 palabras y
  estructural), cuatro variantes de embeddings (`multilingual-e5-large` con y sin prefijos,
  `multilingual-e5-base` y `bge-m3`), con y sin re-ranking (`bge-reranker-v2-m3`), y dos
  generadores (`llama3.2:3b` y `gemma3:4b`).
- **Resultado:** la mejor configuración es fragmentación estructural + `bge-m3` + re-ranking
  con k = 3, que deja la evidencia entre los fragmentos recuperados en el 95,7 % de las
  preguntas (100 % con k = 5). Con `gemma3:4b` y el umbral de abstención, el asistente cita
  un artículo correcto en el 87,0 % de las preguntas del código, rechaza las 6 preguntas
  externas y responde en unos 3 s en una T4.

  | Hipótesis                              | Resultado               |
  | -------------------------------------- | ----------------------- |
  | H1. Fragmentación estructural          | Se cumple               |
  | H2. Prefijos `query:`/`passage:` de E5 | No se cumple            |
  | H3. Re-ranking mejora el MRR           | Se cumple               |
  | H4. `gemma3:4b` cita mejor             | Se cumple               |
  | H5. Umbral de abstención               | Se cumple, con reservas |

### Taller-s6 — Amazon Review Intelligence: asistente sobre reseñas con MCP y Skills

Asistente conversacional en español que responde preguntas sobre reseñas de Amazon:
localiza testimonios relevantes, calcula cifras exactas e interpreta opiniones citando los
IDs de las reseñas en que se apoya. Todo corre en local, sin servicios de inferencia en la
nube.

```
Gradio → agente Python + Qwen3 → MCP → búsqueda FAISS o estadísticas con pandas → auditoría → respuesta con evidencia
```

- **Cuaderno:** [Tallser-s6/01_Amazon_Review_Intelligence.ipynb](Tallser-s6/01_Amazon_Review_Intelligence.ipynb),
  que prepara el corpus y el índice si faltan, levanta y verifica el servidor MCP, carga las
  Skills, ejecuta sus pruebas y abre el chat. Los scripts están en
  [Tallser-s6/Amazon_Review_Intelligence_Entrega/](Tallser-s6/Amazon_Review_Intelligence_Entrega/).
- **Corpus:** muestra balanceada de 1.000 reseñas de la partición `train` de
  `SetFit/amazon_reviews_multi_es` (200 por calificación, semilla 42, sin IDs ni textos
  repetidos), guardada en `data/resenas.csv`. Los porcentajes que calcula el asistente
  describen solo esta muestra, no la distribución real de Amazon.
- **Modelos:** `qwen3:4b-instruct-2507-q4_K_M` como generador (temperatura 0, contexto de
  4.096 tokens) y `embeddinggemma:300m-qat-q4_0` para los embeddings (768 dimensiones),
  ambos servidos con Ollama. El índice es un `IndexFlatIP` de FAISS sobre vectores
  normalizados, es decir, similitud coseno.
- **Herramientas MCP:** un servidor FastMCP sobre HTTP local (`127.0.0.1:8000/mcp`) publica
  dos herramientas:

  | Herramienta            | Qué hace                                                                                         |
  | ---------------------- | ------------------------------------------------------------------------------------------------ |
  | `buscar_resenas`       | Recupera hasta 8 reseñas por similitud semántica, con filtro opcional por estrellas.             |
  | `estadisticas_resenas` | Cuenta reseñas con pandas por estrellas y palabra clave literal; devuelve porcentaje y promedio. |

- **Skills:** dos procedimientos en `SKILL.md` que no son herramientas MCP adicionales.
  `analisis-voz-consumidor` guía la interpretación de opiniones sin añadir detalles
  ausentes y `auditoria-evidencia-rag` fija los controles de trazabilidad. El gestor
  (`08_gestor_skills.py`) activa una, otra o ambas según la pregunta pida ejemplos o cifras,
  y verifica de forma determinista que los IDs existan, que las citas coincidan con el CSV y
  que los porcentajes respeten los filtros.
- **Ajustes durante el desarrollo:** respuesta directa a saludos sin consultar FAISS,
  separación entre filtrar por estrellas y pedir un conteo, corrección de argumentos mal
  tipados por Qwen (`estrellas: 4` → `[4]`), resolución de referencias como «esa misma
  reseña» a partir del historial y control de síntesis que contradicen un conteo de MCP.
- **Límites:** solo 1.000 reseñas; la búsqueda semántica no garantiza cubrir todas las
  opiniones; los documentos recuperados y las coincidencias literales no son el mismo
  subconjunto; el historial resuelve referencias sencillas, no memoria general. Los
  controles automáticos comprueban IDs, textos, calificaciones y aritmética, pero la
  relevancia semántica y la calidad de las paráfrasis requieren revisión humana.
- **Salidas:** `data/resenas.csv` y `data/origen.json` con la muestra y sus metadatos,
  `vectorstore/resenas.faiss` y `vectorstore/config.json` con el índice, y
  `mcp_servidor.log` cuando el servidor se lanza desde el cuaderno, todo dentro de
  `Amazon_Review_Intelligence_Entrega/`.

## Datos

Los corpus **no se guardan en el repositorio**. Los cuadernos los descargan del Hub de
Hugging Face con `load_dataset` en la primera ejecución y quedan en la caché local de la
librería (`~/.cache/huggingface/datasets`); no hay que preparar nada a mano. La carpeta
`data/` se reserva para archivos de apoyo pequeños y propios de cada taller. El cuaderno de
la semana 5 lee de ahí el texto limpio de la Ley 769 de 2002 y su set de evaluación; si no
los encuentra en local, los descarga de este repositorio en GitHub.

## Requisitos

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```

Los cuadernos de las semanas 1 y 2 detectan si se ejecutan en Google Colab e instalan las
dependencias por su cuenta. Funcionan en CPU.

El cuaderno de la semana 3 está pensado para Google Colab con GPU (se ejecutó en una L4). Su
primera celda instala versiones fijas de `transformers` (4.57.1), `datasets` (4.3.0),
`accelerate`, `huggingface-hub` y `tokenizers`, y exige PyTorch 2.6 o superior; después hay
que reiniciar la sesión una vez. En CPU funciona, pero el entrenamiento de los tres modelos
tarda mucho más.

El cuaderno de la semana 4 instala sus dependencias en la primera celda y detecta el
dispositivo disponible (CUDA, MPS o CPU). Se ejecutó con MPS en un equipo Apple Silicon,
donde los dos entrenamientos tardan unos siete minutos en total.

El cuaderno de la semana 5 está pensado para Google Colab con una GPU T4. Instala versiones
fijas de LangChain, FAISS, `sentence-transformers`, `ollama` y Gradio, instala Ollama,
arranca su servidor en segundo plano y descarga `llama3.2:3b` y `gemma3:4b`. Conviene
empezar con un entorno de ejecución nuevo para que la GPU tenga memoria libre para los
embeddings, el re-ranker y Ollama a la vez. Fuera de Colab hay que instalar Ollama a mano
desde https://ollama.com/download.

El taller de la semana 6 se ejecuta en local (se desarrolló en Windows 11 con Python 3.12,
una NVIDIA GTX 1660 SUPER de 6 GB y 8 GB de RAM) y tiene sus propias dependencias, con
Gradio fijado en 5.50.0. Requiere Ollama en marcha en `127.0.0.1:11434` con los dos
modelos descargados:

```bash
pip install -r Tallser-s6/Amazon_Review_Intelligence_Entrega/requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull embeddinggemma:300m-qat-q4_0
```

## Ejecución

Abrir la carpeta del taller correspondiente y ejecutar las celdas en orden. El cuaderno de
la semana 1 entrena siete modelos, evalúa el mejor sobre la partición de prueba y levanta
una demo en Gradio al final. El de la semana 2 entrena diecisiete modelos y sigue el mismo
cierre. El de la semana 3 afina los tres modelos BERT uno tras otro, evalúa cada uno sobre la
partición de prueba y termina con la tabla comparativa y el análisis de errores. El de la
semana 4 entrena los dos Mini-GPT, evalúa el mejor sobre la partición de prueba, compara
cinco estrategias de decodificación y deja la función `demo_interactiva()` para probar un
título propio. El de la
semana 5 corre los 24 experimentos de recuperación, compara los dos generadores, calibra el
umbral y levanta el chatbot en Gradio con un enlace público temporal (`share=True` en Colab).

El cuaderno de la semana 6 toma como raíz el directorio de trabajo, así que debe ejecutarse
con `Amazon_Review_Intelligence_Entrega/` como directorio actual. Reutiliza el corpus y el
índice si ya existen, arranca el servidor MCP si el puerto 8000 está libre y abre el chat en
http://127.0.0.1:7860. Sin el cuaderno, los scripts se lanzan en este orden:

```bash
cd Tallser-s6/Amazon_Review_Intelligence_Entrega
python 01_preparar_resenas.py        # data/resenas.csv
python 02_construir_indice.py        # vectorstore/resenas.faiss
python 09_probar_skills.py           # pruebas del gestor; no necesita Ollama ni MCP
python 04_servidor_mcp_v3_2.py       # dejar en marcha (puerto 8000)
python 07_app_gradio_v4_skills.py    # en otra terminal (puerto 7860)
```

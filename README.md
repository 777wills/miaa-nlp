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
├── data/              # archivos de apoyo locales (los corpus grandes no se versionan)
├── requirements.txt   # dependencias comunes a todos los talleres
├── Taller-s1/         # semana 1
│   └── clasificacion_resenas_es_lstm_gru.ipynb
├── Taller-s2/         # semana 2
│   └── clasificacion_resenas_es_transformers.ipynb
├── Taller-s3/         # semana 3
│   └── Proyecto_bert_resenas_amazon_colab_Experimentos.ipynb
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

## Datos

Los corpus **no se guardan en el repositorio**. Los cuadernos los descargan del Hub de
Hugging Face con `load_dataset` en la primera ejecución y quedan en la caché local de la
librería (`~/.cache/huggingface/datasets`); no hay que preparar nada a mano. La carpeta
`data/` se reserva para archivos de apoyo pequeños y propios de cada taller.

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

## Ejecución

Abrir la carpeta del taller correspondiente y ejecutar las celdas en orden. El cuaderno de
la semana 1 entrena siete modelos, evalúa el mejor sobre la partición de prueba y levanta
una demo en Gradio al final. El de la semana 2 entrena diecisiete modelos y sigue el mismo
cierre. El de la semana 3 afina los tres modelos BERT uno tras otro, evalúa cada uno sobre la
partición de prueba y termina con la tabla comparativa y el análisis de errores.

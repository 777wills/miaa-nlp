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

Los cuadernos detectan si se ejecutan en Google Colab e instalan las dependencias por su
cuenta. Funcionan en CPU, aunque el entrenamiento de las redes es notablemente más rápido
con GPU.

## Ejecución

Abrir la carpeta del taller correspondiente y ejecutar las celdas en orden. El cuaderno de
la semana 1 entrena siete modelos, evalúa el mejor sobre la partición de prueba y levanta
una demo en Gradio al final.

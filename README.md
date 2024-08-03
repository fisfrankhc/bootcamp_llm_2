# TAREA 04 - FRANK HUAMAN

## Descripción

Este proyecto utiliza el modelo BERT para generar embeddings y realizar fine-tuning en un conjunto de datos de ventas. El objetivo de este entrenamiento previo es clasificar las ventas en completadas y no completadas, y eventualmente utilizar estos embeddings para desarrollar un chatbot.

## Modelo Encoder Utilizado

Se utilizó el modelo `bert-base-uncased` de BERT, un modelo preentrenado de la biblioteca `transformers` de Hugging Face. Este modelo es capaz de generar representaciones numéricas (embeddings) de textos, que son útiles para diversas tareas de procesamiento de lenguaje natural (NLP).

## Propósito del Proyecto

1. **Generación de Embeddings**: Utilizar BERT para generar embeddings de textos descriptivos de las ventas.
2. **Fine-Tuning**: Entrenar un modelo de clasificación para identificar ventas completadas y no completadas.
3. **Desarrollo de Chatbot**: Utilizar los embeddings y el modelo entrenado como base para un chatbot que interactúe con el sistema de ventas.

## Requisitos

- Python 3.8 o superior
- Bibliotecas: `pandas`, `torch`, `transformers`, `scikit-learn`

## Instalación

1. Clona el repositorio:
    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    ```
2. Navega al directorio del proyecto:
    ```bash
    cd tu_repositorio
    ```
3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución del Proyecto

   MÉTODO 1:
      1. Ingresamos a la carpeta del proyecto y ubicamos el archivo `principal.ipynb`
      2. Abrimos el archivo.
      3. Verificamos que contamos con las librerias establecidas. En caso no, lo instalamos usando `%pip install`
      4. Ejecutamos según el orden del codigo.

   MÉTODO 2 (ALTERNO):
      5. Carga y procesa los datos:
          ```python
          ventas, ventas_detalle, sucursales, usuarios, productos = cargar_datos()
          df_procesado = procesar_datos(ventas, ventas_detalle, sucursales, usuarios, productos)
          ```

      6. Genera embeddings y entrena el modelo:
          ```python
          main()
          ```

## Estructura del Proyecto

```plaintext
.
├── databd
│   ├── farmacia_ventas.csv
│   ├── farmacia_venta_detalle.csv
│   ├── general_sucursales.csv
│   ├── general_usuarios.csv
│   └── producto_productos.csv
├── results
├── logs
├── main.py
├── principal.ipynb
├── requirements.txt
└── README.md

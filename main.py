import pandas as pd
from typing import List, Tuple
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

def cargar_datos() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos desde archivos CSV.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        Tuple con cinco DataFrames: ventas, ventas_detalle, sucursales, usuarios, productos.
    """
    ventas = pd.read_csv('databd/farmacia_ventas.csv')
    ventas_detalle = pd.read_csv('databd/farmacia_venta_detalle.csv')
    sucursales = pd.read_csv('databd/general_sucursales.csv')
    usuarios = pd.read_csv('databd/general_usuarios.csv')
    productos = pd.read_csv('databd/producto_productos.csv')
    return ventas, ventas_detalle, sucursales, usuarios, productos

def procesar_datos(ventas: pd.DataFrame, ventas_detalle: pd.DataFrame, 
                   sucursales: pd.DataFrame, usuarios: pd.DataFrame, 
                   productos: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa y une múltiples DataFrames en un único DataFrame consolidado.

    Args:
        ventas (pd.DataFrame): DataFrame con los datos de ventas.
        ventas_detalle (pd.DataFrame): DataFrame con los detalles de las ventas.
        sucursales (pd.DataFrame): DataFrame con la información de las sucursales.
        usuarios (pd.DataFrame): DataFrame con la información de los usuarios.
        productos (pd.DataFrame): DataFrame con la información de los productos.

    Returns:
        pd.DataFrame: DataFrame consolidado.
    """
    ventas['venta_id'] = ventas['venta_id'].astype(str)
    ventas_detalle['venta_id'] = ventas_detalle['venta_id'].astype(str)
    ventas['usuario_id'] = ventas['usuario_id'].astype(str)
    usuarios['user_id'] = usuarios['user_id'].astype(str)
    ventas['sucursal_id'] = ventas['sucursal_id'].astype(str)
    sucursales['suc_id'] = sucursales['suc_id'].astype(str)
    ventas_detalle['prod_id'] = ventas_detalle['prod_id'].astype(str)
    productos['prod_id'] = productos['prod_id'].astype(str)

    df = ventas.merge(ventas_detalle, on='venta_id', how='inner')
    df = df.merge(sucursales, left_on='sucursal_id', right_on='suc_id', how='inner')
    df = df.merge(usuarios, left_on='usuario_id', right_on='user_id', how='inner')
    df = df.merge(productos, on='prod_id', how='inner')
    
    return df

def preparar_textos(df: pd.DataFrame) -> List[str]:
    """
    Prepara una lista de textos a partir de un DataFrame consolidado.

    Args:
        df (pd.DataFrame): DataFrame consolidado.

    Returns:
        List[str]: Lista de textos preparados para el modelo.
    """
    textos = df.apply(lambda row: f"Producto: {row['prod_nombre']}, Sucursal: {row['suc_nombre']}, Vendedor: {row['user_name']}", axis=1)
    return textos.tolist()

class BertEmbedder:
    """
    Clase para generar embeddings utilizando BERT.
    """
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Inicializa el BertEmbedder con un modelo y tokenizer preentrenados.

        Args:
            model_name (str, optional): Nombre del modelo BERT preentrenado. Por defecto es 'bert-base-uncased'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
    
    def generar_embeddings(self, textos: List[str]) -> torch.Tensor:
        """
        Genera embeddings para una lista de textos utilizando BERT.

        Args:
            textos (List[str]): Lista de textos para los cuales generar embeddings.

        Returns:
            torch.Tensor: Tensor de embeddings generados por BERT.
        """
        inputs = self.tokenizer(textos, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings

class VentasDataset(Dataset):
    """
    Clase para manejar los datos de ventas en un conjunto de datos.

    Args:
        encodings (dict): Diccionario con las codificaciones de los textos.
        labels (List[int]): Lista de etiquetas correspondientes a los textos.
    """
    def __init__(self, encodings: dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        """
        Obtiene un elemento del conjunto de datos.

        Args:
            idx (int): Índice del elemento a obtener.

        Returns:
            dict: Diccionario con las codificaciones y la etiqueta correspondiente.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """
        Obtiene el tamaño del conjunto de datos.

        Returns:
            int: Número de elementos en el conjunto de datos.
        """
        return len(self.labels)

def main():
    # Cargar y procesar los datos
    ventas, ventas_detalle, sucursales, usuarios, productos = cargar_datos()
    df_procesado = procesar_datos(ventas, ventas_detalle, sucursales, usuarios, productos)

    # Crear la columna 'label' con valores 1 si la venta fue pagada y 0 si no
    df_procesado['label'] = df_procesado['venta_proceso'].apply(lambda x: 1 if x.strip().lower() == 'pagado' else 0)

    # Verificar los valores de la columna 'label'
    print(df_procesado[['venta_proceso', 'label']].sample(15))

    textos = preparar_textos(df_procesado)

    # Generar embeddings
    embedder = BertEmbedder()
    embeddings = embedder.generar_embeddings(textos)
    print(embeddings.shape)

    # Tokenización y preparación de datos
    train_texts, val_texts, train_labels, val_labels = train_test_split(textos, df_procesado['label'].tolist(), test_size=0.2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = VentasDataset(train_encodings, train_labels)
    val_dataset = VentasDataset(val_encodings, val_labels)

    # Configurar el modelo de clasificación
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    # Definir los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="no",
    )

    # Crear el entrenador
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Entrenar el modelo
    trainer.train()
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
"""Treinamento da MLP NIDS usando arquivos PCA."""

import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from .config import ATTACK_LABEL, CUSTOM_CALLBACK, NORMAL_LABEL, SEED, TARGET
from .s3 import conectar_duckdb_s3, join_s3_path, sql_path, upload_file_to_s3


class NIDSCallback(Callback):
    """Calcula métricas essenciais para validação em datasets desbalanceados.

    Métricas calculadas:
    - macro_f1;
    - balanced_accuracy;
    - f1_normal;
    - f1_attack;
    - recall_normal;
    - recall_attack.
    """

    def __init__(
        self,
        validation_data: tuple[pd.DataFrame, np.ndarray],
        normal_label: int = 0,
        attack_label: int = 1,
        prefix: str = "val",
        batch_size: int = 1024,
    ) -> None:
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.normal_label = normal_label
        self.attack_label = attack_label
        self.prefix = prefix
        self.batch_size = batch_size

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        """Calcula métricas ao final de cada época.

        Args:
            epoch: Índice da época finalizada.
            logs: Dicionário de métricas mantido pelo Keras. A função adiciona
                as métricas customizadas nele.
        """
        logs = logs or {}

        y_prob = self.model.predict(
            self.x_val,
            batch_size=self.batch_size,
            verbose=0,
        )

        y_pred = np.argmax(y_prob, axis=1)

        macro_f1 = f1_score(
            self.y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )

        balanced_accuracy = balanced_accuracy_score(
            self.y_val,
            y_pred,
        )

        f1_normal = f1_score(
            self.y_val,
            y_pred,
            pos_label=self.normal_label,
            zero_division=0,
        )

        f1_attack = f1_score(
            self.y_val,
            y_pred,
            pos_label=self.attack_label,
            zero_division=0,
        )

        recall_normal = recall_score(
            self.y_val,
            y_pred,
            pos_label=self.normal_label,
            zero_division=0,
        )

        recall_attack = recall_score(
            self.y_val,
            y_pred,
            pos_label=self.attack_label,
            zero_division=0,
        )

        logs[f"{self.prefix}_macro_f1"] = float(macro_f1)
        logs[f"{self.prefix}_balanced_accuracy"] = float(balanced_accuracy)
        logs[f"{self.prefix}_f1_normal"] = float(f1_normal)
        logs[f"{self.prefix}_f1_attack"] = float(f1_attack)
        logs[f"{self.prefix}_recall_normal"] = float(recall_normal)
        logs[f"{self.prefix}_recall_attack"] = float(recall_attack)

        print(
            f"\n[{self.prefix.upper()} METRICS] "
            f"epoch={epoch + 1} | "
            f"macro_f1={macro_f1:.4f} | "
            f"balanced_acc={balanced_accuracy:.4f} | "
            f"f1_normal={f1_normal:.4f} | "
            f"f1_attack={f1_attack:.4f} | "
            f"recall_normal={recall_normal:.4f} | "
            f"recall_attack={recall_attack:.4f}",
            flush=True,
        )


@dataclass(frozen=True)
class ModelConfig:
    """Configuração do treino usando entradas PCA.

    Attributes:
        batch_size: Tamanho dos batches do treinamento Keras.
        epochs: Quantidade máxima de épocas.
        learning_rate: Taxa de aprendizado usada pelo modelo definido no notebook.
        early_stop_patience: Paciência do EarlyStopping.
        random_seed: Seed usada para reprodutibilidade.
        clean_local_artifacts: Remove artefatos temporários locais ao final do treino.
    """

    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 0.001
    early_stop_patience: int = 5
    random_seed: int = SEED
    clean_local_artifacts: bool = True


def pca_paths(dataset: dict[str, str]) -> tuple[str, str, str]:
    """Monta os caminhos dos arquivos PCA de treino, validação e teste.

    Args:
        dataset: Configuração do dataset com `DATASET_KEY` e `SPLIT_PATH`.

    Returns:
        Caminhos dos arquivos PCA de treino, validação e teste.
    """
    dataset_key = dataset["DATASET_KEY"]
    split_path = dataset["SPLIT_PATH"]

    train_pca = join_s3_path(split_path, f"{dataset_key}_train_pca.parquet")
    val_pca = join_s3_path(split_path, f"{dataset_key}_val_pca.parquet")
    test_pca = join_s3_path(split_path, f"{dataset_key}_test_pca.parquet")

    return train_pca, val_pca, test_pca


def artifact_path(dataset: dict[str, str]) -> str:
    """Retorna o prefixo S3 para artefatos do treino.

    Args:
        dataset: Configuração do dataset com `ARTIFACT_PATH`.

    Returns:
        Prefixo S3 onde os artefatos do treino serão salvos.
    """
    return join_s3_path(dataset["ARTIFACT_PATH"], "result")


def parquet_to_df(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> pd.DataFrame:
    """Lê um arquivo Parquet do S3 usando DuckDB.

    Args:
        con: Conexão DuckDB configurada para S3.
        parquet_path: Caminho do arquivo Parquet.

    Returns:
        DataFrame Pandas com os dados.
    """
    print(f"Lendo: {parquet_path}", flush=True)

    return con.execute(
        f"""
        SELECT *
        FROM read_parquet('{sql_path(parquet_path)}')
        """
    ).fetchdf()


def separar_df(
    df: pd.DataFrame,
    target: str = "Label",
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Separa features e target.

    Args:
        df: DataFrame contendo features e target.
        target: Nome da coluna alvo.
        feature_columns: Lista opcional de colunas de entrada.
            Quando informada, força a mesma ordem das features.

    Returns:
        Tupla com X e y.
    """
    if target not in df.columns:
        raise ValueError(f"Coluna target não encontrada: {target}")

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target]

    x = df[feature_columns].astype("float32")
    y = df[target].astype("int32").to_numpy()

    return x, y


def mostrar_distribuicao(
    y: np.ndarray,
    name: str,
) -> None:
    """Mostra a distribuição de labels.

    Args:
        y: Array com os rótulos do split.
        name: Nome exibido no log para identificar o split.
    """
    labels, counts = np.unique(y, return_counts=True)

    distribution = {int(label): int(count) for label, count in zip(labels, counts)}

    total = int(len(y))

    print(f"Distribuição {name}: {distribution} | total={total:,}", flush=True)


def calcular_pesos(
    y_train: np.ndarray,
) -> dict[int, float]:
    """Calcula class_weight balanceado para as classes 0 e 1.

    Args:
        y_train: Labels do treino.

    Returns:
        Dicionário no formato aceito pelo Keras.
    """
    unique_labels = set(np.unique(y_train).astype(int).tolist())

    required_labels = {NORMAL_LABEL, ATTACK_LABEL}

    if not required_labels.issubset(unique_labels):
        raise ValueError(
            "O treino precisa conter as classes 0 e 1. "
            f"Classes encontradas: {sorted(unique_labels)}"
        )

    classes = np.array([NORMAL_LABEL, ATTACK_LABEL])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {
        NORMAL_LABEL: float(weights[0]),
        ATTACK_LABEL: float(weights[1]),
    }


def json_default(value: Any) -> Any:
    """Converte tipos NumPy para tipos nativos do Python no JSON.

    Args:
        value: Valor recebido pelo serializador JSON.

    Returns:
        Valor convertido para um tipo serializável pelo módulo `json`.

    Raises:
        TypeError: Quando o valor não tem conversão conhecida.
    """
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(f"Tipo não serializável: {type(value)}")


def salvar_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Salva dicionário em JSON.

    Args:
        data: Dados a gravar.
        path: Caminho do arquivo JSON local.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


def upload_outputs(local_dir: Path, artifact_prefix: str) -> None:
    """Envia apenas os artefatos essenciais do treino para o bucket.

    Args:
        local_dir: Diretório local onde os artefatos foram gravados.
        artifact_prefix: Prefixo S3 de destino.
    """
    files = [
        "best_model.keras",       # melhor modelo pelo macro_f1
        "training_history.json",   # historico
        ]

    for file_name in files:
        local_file = local_dir / file_name
        s3_uri = join_s3_path(artifact_prefix, file_name)

        upload_file_to_s3(
            local_path=local_file,
            s3_uri=s3_uri,
        )


def treinar_dataset(
    dataset: dict[str, str],
    config: ModelConfig,
    model_builder: Callable[[int, ModelConfig], tf.keras.Model],
) -> None:
    """Treina o modelo usando arquivos PCA e salva o melhor resultado no S3.

    A arquitetura do modelo é recebida por `model_builder` para que as camadas
    fiquem visíveis e parametrizáveis no notebook.

    Args:
        dataset: Configuração do dataset com paths S3 e identificadores.
        config: Parâmetros de treinamento.
        model_builder: Função que recebe `input_dim` e `config` e retorna um
            modelo Keras compilado.
    """
    dataset_key = dataset["DATASET_KEY"]
    train_pca, val_pca, _ = pca_paths(dataset)
    artifact_prefix = artifact_path(dataset)

    local_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_key}_treino_pca_"))
    con = conectar_duckdb_s3()

    try:
        print("=" * 80, flush=True)
        print(f"Treino: {dataset_key}", flush=True)

        # 1. Carrega Treino e Validação
        train_df = parquet_to_df(con, train_pca)
        val_df = parquet_to_df(con, val_pca)

        x_train, y_train = separar_df(df=train_df, target=TARGET)
        feature_columns = x_train.columns.tolist()

        x_val, y_val = separar_df(
            df=val_df,
            target=TARGET,
            feature_columns=feature_columns
        )

        del train_df, val_df # Liberando RAM

        mostrar_distribuicao(y_train, "treino")
        mostrar_distribuicao(y_val, "validação")

        class_weight = calcular_pesos(y_train)
        input_dim = x_train.shape[1]

        # 2. Inicializa o modelo definido pelo notebook
        model = model_builder(input_dim, config)

        best_model_path = local_dir / "best_model.keras"
        final_model_path = local_dir / "model.keras"

        # Callbacks monitorando val_macro_f1 para salvar o melhor peso
        callbacks = [
            NIDSCallback(
                validation_data=(x_val, y_val),
                normal_label=NORMAL_LABEL,
                attack_label=ATTACK_LABEL,
                prefix="val",
                batch_size=2048,
            ),
            EarlyStopping(
                monitor=CUSTOM_CALLBACK,
                mode="max",
                patience=config.early_stop_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=best_model_path,
                monitor=CUSTOM_CALLBACK,
                mode="max",
                save_best_only=True,
                verbose=1,
            )
        ]

        # 3. Executa o treinamento
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        # Salva o modelo da última época por segurança
        model.save(final_model_path)

        # Salva o histórico de treino localmente
        salvar_json(data=history.history, path=local_dir / "training_history.json")

        # 4. Sobe os artefatos de treino para o S3 (Envia o best_model.keras)
        upload_outputs(local_dir=local_dir, artifact_prefix=artifact_prefix)
        print(f"Treinamento concluído e artefatos salvos para {dataset_key}!", flush=True)
        print("=" * 80, flush=True)

    finally:
        con.close()
        if config.clean_local_artifacts:
            shutil.rmtree(local_dir, ignore_errors=True)

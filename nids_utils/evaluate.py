"""Avaliação in-domain e cross-domain dos modelos NIDS."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import ATTACK_LABEL, NORMAL_LABEL, TARGET
from .preprocess import aplicar_pipeline_cross_domain
from .s3 import (
    conectar_duckdb_s3,
    download_file_from_s3,
    join_s3_path,
    upload_file_to_s3,
)
from .train import artifact_path, parquet_to_df, pca_paths, separar_df


def json_default(value: Any) -> Any:
    """Converte tipos NumPy para tipos nativos serializáveis."""
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(f"Tipo não serializável: {type(value)}")


def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Calcula métricas principais para NIDS binário."""
    labels = [NORMAL_LABEL, ATTACK_LABEL]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_attack": precision_score(
            y_true,
            y_pred,
            pos_label=ATTACK_LABEL,
            zero_division=0,
        ),
        "recall_attack": recall_score(
            y_true,
            y_pred,
            pos_label=ATTACK_LABEL,
            zero_division=0,
        ),
        "f1_attack": f1_score(
            y_true,
            y_pred,
            pos_label=ATTACK_LABEL,
            zero_division=0,
        ),
        "precision_normal": precision_score(
            y_true,
            y_pred,
            pos_label=NORMAL_LABEL,
            zero_division=0,
        ),
        "recall_normal": recall_score(
            y_true,
            y_pred,
            pos_label=NORMAL_LABEL,
            zero_division=0,
        ),
        "f1_normal": f1_score(
            y_true,
            y_pred,
            pos_label=NORMAL_LABEL,
            zero_division=0,
        ),
        "fpr": fpr,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "total": int(len(y_true)),
    }


def baixar_modelo(
    model_s3_uri: str,
    local_dir: Path,
) -> Path:
    """Baixa um modelo Keras do S3 para avaliação local."""
    local_model = local_dir / Path(model_s3_uri).name
    download_file_from_s3(model_s3_uri, local_model)
    return local_model


def salvar_resultado(
    resultado: dict[str, Any],
    output_path: str,
    file_name: str,
) -> str:
    """Salva o resultado da avaliação em JSON no S3."""
    local_dir = Path(tempfile.mkdtemp(prefix="nids_eval_result_"))
    s3_uri = join_s3_path(output_path, file_name)
    resultado["result_path"] = s3_uri

    try:
        local_file = local_dir / file_name

        with local_file.open("w", encoding="utf-8") as file:
            json.dump(
                resultado,
                file,
                ensure_ascii=False,
                indent=2,
                default=json_default,
            )

        upload_file_to_s3(local_file, s3_uri)
        return s3_uri

    finally:
        shutil.rmtree(local_dir, ignore_errors=True)


def avaliar_arquivo_pca(
    train_dataset: dict[str, str],
    test_dataset: dict[str, str],
    test_pca_path: str,
    output_path: str,
    evaluation_type: str,
    batch_size: int = 2048,
    target: str = TARGET,
) -> dict[str, Any]:
    """Avalia um modelo treinado em um arquivo PCA de teste."""
    train_key = train_dataset["DATASET_KEY"]
    test_key = test_dataset["DATASET_KEY"]
    model_s3_uri = join_s3_path(artifact_path(train_dataset), "best_model.keras")

    local_dir = Path(tempfile.mkdtemp(prefix=f"{train_key}_eval_"))
    con = conectar_duckdb_s3()

    try:
        local_model = baixar_modelo(model_s3_uri, local_dir)
        model = tf.keras.models.load_model(local_model)

        test_df = parquet_to_df(con, test_pca_path)
        x_test, y_test = separar_df(df=test_df, target=target)
        del test_df

        y_prob = model.predict(
            x_test,
            batch_size=batch_size,
            verbose=0,
        )
        y_pred = np.argmax(y_prob, axis=1)

        metrics = calcular_metricas(y_test, y_pred)

        resultado = {
            "evaluation_type": evaluation_type,
            "train_dataset": train_key,
            "test_dataset": test_key,
            "model_path": model_s3_uri,
            "test_pca_path": test_pca_path,
            "metrics": metrics,
        }

        file_name = f"{train_key}_to_{test_key}_{evaluation_type}_metrics.json"
        salvar_resultado(
            resultado=resultado,
            output_path=output_path,
            file_name=file_name,
        )

        print("Avaliação concluída:", flush=True)
        print(f"Treino: {train_key}", flush=True)
        print(f"Teste: {test_key}", flush=True)
        print(f"Tipo: {evaluation_type}", flush=True)
        print(f"macro_f1: {metrics['macro_f1']:.4f}", flush=True)
        print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}", flush=True)
        print(f"recall_attack: {metrics['recall_attack']:.4f}", flush=True)
        print(f"fpr: {metrics['fpr']:.4f}", flush=True)
        print(f"Resultado: {resultado['result_path']}", flush=True)
        print("=" * 80, flush=True)

        return resultado

    finally:
        con.close()
        shutil.rmtree(local_dir, ignore_errors=True)


def avaliar_in_domain(
    dataset: dict[str, str],
    output_path: str,
    batch_size: int = 2048,
    target: str = TARGET,
) -> dict[str, Any]:
    """Avalia o modelo no test PCA do mesmo dataset."""
    _, _, test_pca = pca_paths(dataset)

    return avaliar_arquivo_pca(
        train_dataset=dataset,
        test_dataset=dataset,
        test_pca_path=test_pca,
        output_path=output_path,
        evaluation_type="in_domain",
        batch_size=batch_size,
        target=target,
    )


def avaliar_cross_domain(
    train_dataset: dict[str, str],
    test_dataset: dict[str, str],
    cross_output_path: str,
    result_output_path: str,
    numeric_features: list[str],
    categorical_features: list[str],
    batch_size: int = 2048,
    pca_batch_size: int = 100_000,
    target: str = TARGET,
) -> dict[str, Any]:
    """Gera o PCA cross correto e avalia o modelo treinado em outro domínio."""
    _, cross_pca = aplicar_pipeline_cross_domain(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output_path=cross_output_path,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target=target,
        split_name="test",
        batch_size=pca_batch_size,
    )

    return avaliar_arquivo_pca(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        test_pca_path=cross_pca,
        output_path=result_output_path,
        evaluation_type="cross_domain",
        batch_size=batch_size,
        target=target,
    )

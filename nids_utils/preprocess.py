"""Pipeline de preprocessamento dos datasets NIDS."""

import json
import math
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA

from .s3 import (
    conectar_duckdb_s3,
    download_file_from_s3,
    join_s3_path,
    sql_path,
    to_string,
    upload_file_to_s3,
    validar_objeto_s3,
)


def calcular_estatisticas_treino(
    con: duckdb.DuckDBPyConnection,
    train_path: str,
    numeric_features: list[str],
) -> dict[str, tuple[float, float]]:
    """Calcula média e desvio padrão usando somente o treino.

    Args:
        con: Conexão DuckDB configurada para ler o arquivo de treino.
        train_path: Caminho do Parquet de treino.
        numeric_features: Colunas numéricas usadas no cálculo.

    Returns:
        Dicionário no formato `{coluna: (media, desvio_padrao)}`.
    """

    select_expr = ", ".join(
        f"""
        AVG({to_string(col)}) AS {to_string(col + "_mean")},
        STDDEV_POP({to_string(col)}) AS {to_string(col + "_std")}
        """
        for col in numeric_features
    )

    row = con.execute(
        f"""
        SELECT {select_expr}
        FROM read_parquet('{sql_path(train_path)}')
        """
    ).fetchone()

    stats = {}

    idx = 0
    for col in numeric_features:
        mean = row[idx]
        std = row[idx + 1]

        if mean is None:
            mean = 0.0

        if std is None or std == 0:
            std = 1.0

        stats[col] = (float(mean), float(std))
        idx += 2

    return stats


def count_rows(con: duckdb.DuckDBPyConnection, file_path: str) -> int:
    """Conta as linhas de um arquivo Parquet.

    Args:
        con: Conexão DuckDB configurada.
        file_path: Caminho do arquivo Parquet.

    Returns:
        Quantidade de linhas encontradas.
    """
    return con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_path(file_path)}')
        """
    ).fetchone()[0]


def count_rows_label(
    con: duckdb.DuckDBPyConnection,
    file_path: str,
    split_name: str,
    target: str = "Label",
) -> None:
    """Exibe a quantidade de linhas por classe em um split.

    Args:
        con: Conexão DuckDB configurada.
        file_path: Caminho do arquivo Parquet.
        split_name: Nome exibido no log para identificar o split.
        target: Nome da coluna alvo.
    """
    rows = con.execute(
        f"""
        SELECT
            {to_string(target)},
            COUNT(*) AS total
        FROM read_parquet('{sql_path(file_path)}')
        GROUP BY {to_string(target)}
        ORDER BY {to_string(target)}
        """
    ).fetchall()

    print(f"{split_name}: {rows}")


def criar_split_hash(
    con: duckdb.DuckDBPyConnection,
    source_file: str,
    target_file: str,
    split_name: str,
    bucket_start: int,
    bucket_end: int,
    hash_columns: list[str],
    random_seed: int = 42,
) -> None:
    """Cria um split por hash determinístico com baixo uso de memória.

    Args:
        con: Conexão DuckDB.
        source_file: Arquivo Parquet de origem.
        target_file: Arquivo Parquet de saída.
        split_name: Nome do split exibido no log.
        bucket_start: Bucket inicial inclusivo.
        bucket_end: Bucket final exclusivo.
        hash_columns: Colunas usadas para gerar o hash determinístico.
        random_seed: Seed adicionada ao hash.
    """
    if not hash_columns:
        raise ValueError("hash_columns não pode ser vazio.")

    hash_expr = ", ".join(to_string(col) for col in hash_columns)

    sql = f"""
    COPY (
        WITH base AS (
            SELECT
                *,
                abs(hash({hash_expr}, {random_seed})) % 100 AS __bucket
            FROM read_parquet('{sql_path(source_file)}')
        )
        SELECT *
        EXCLUDE (__bucket)
        FROM base
        WHERE __bucket >= {bucket_start}
          AND __bucket < {bucket_end}
    )
    TO '{sql_path(target_file)}'
    (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """

    print(f"Gerando {split_name}: {target_file}")
    con.execute(sql)

    total = count_rows(con, target_file)
    print(f"{split_name}: {total:,} linhas")


def split_dataset(
    con: duckdb.DuckDBPyConnection,
    source_file: str,
    output_path: str,
    dataset_name: str,
    hash_columns: list[str],
    target: str = "Label",
    train_size: float = 0.60,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> tuple[str, str, str]:
    """Divide um dataset em treino, validação e teste usando hash determinístico.

    Essa função é adequada para pouca RAM porque:
    - não usa Pandas;
    - não usa row_number;
    - não usa window function;
    - não cria tabela temporária gigante;
    - grava cada split diretamente em Parquet.

    Args:
        con: Conexão DuckDB.
        source_file: Arquivo Parquet limpo de origem.
        output_path: Diretório/prefixo onde os splits serão salvos.
        dataset_name: Nome usado nos arquivos de saída.
        hash_columns: Colunas usadas para gerar o split determinístico.
        target: Nome da coluna alvo.
        train_size: Proporção do treino.
        val_size: Proporção da validação.
        random_seed: Seed usada no hash.

    Returns:
        Caminhos dos arquivos de treino, validação e teste.
    """
    if not 0 < train_size < 1:
        raise ValueError("train_size precisa estar entre 0 e 1.")

    if not 0 <= val_size < 1:
        raise ValueError("val_size precisa estar entre 0 e 1.")

    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size precisa ser menor que 1.")

    if not hash_columns:
        raise ValueError("hash_columns não pode ser vazio.")

    train_end = int(train_size * 100)
    val_end = int((train_size + val_size) * 100)

    train_file = f"{output_path}/{dataset_name}_train.parquet"
    val_file = f"{output_path}/{dataset_name}_val.parquet"
    test_file = f"{output_path}/{dataset_name}_test.parquet"

    print(f"\nDataset: ({output_path}/{dataset_name})")
    print(f"Origem: {source_file}")
    print(
        f"Split: treino={train_size:.0%}, "
        f"validacao={val_size:.0%}, "
        f"teste={1 - train_size - val_size:.0%}"
    )

    total = count_rows(con, source_file)
    print(f"Total limpo: {total:,} linhas")

    criar_split_hash(
        con=con,
        source_file=source_file,
        target_file=train_file,
        split_name="treino",
        bucket_start=0,
        bucket_end=train_end,
        hash_columns=hash_columns,
        random_seed=random_seed,
    )

    criar_split_hash(
        con=con,
        source_file=source_file,
        target_file=val_file,
        split_name="validacao",
        bucket_start=train_end,
        bucket_end=val_end,
        hash_columns=hash_columns,
        random_seed=random_seed,
    )

    criar_split_hash(
        con=con,
        source_file=source_file,
        target_file=test_file,
        split_name="teste",
        bucket_start=val_end,
        bucket_end=100,
        hash_columns=hash_columns,
        random_seed=random_seed,
    )

    print("\nDistribuição por classe:")
    count_rows_label(con, train_file, "treino", target)
    count_rows_label(con, val_file, "validacao", target)
    count_rows_label(con, test_file, "teste", target)
    print("=" * 80)

    return train_file, val_file, test_file


def calcular_categorias_treino(
    con: duckdb.DuckDBPyConnection,
    train_path: str,
    categorical_features: list[str],
) -> dict[str, list[int]]:
    """Coleta categorias distintas usando somente o treino.

    Args:
        con: Conexão DuckDB configurada.
        train_path: Caminho do Parquet de treino.
        categorical_features: Colunas categóricas usadas no one-hot encoding.

    Returns:
        Dicionário no formato `{coluna: [categorias]}`.
    """

    categorias = {}

    for col in categorical_features:
        rows = con.execute(
            f"""
            SELECT DISTINCT {to_string(col)}
            FROM read_parquet('{sql_path(train_path)}')
            WHERE {to_string(col)} IS NOT NULL
            ORDER BY {to_string(col)}
            """
        ).fetchall()

        categorias[col] = [row[0] for row in rows]

    return categorias


def converter_parquet(
    input_path: str,
    output_path: str,
) -> None:
    """Converte um dataset bruto para Parquet comprimido.

    Args:
        input_path: Caminho do dataset bruto.
        output_path: Caminho onde o arquivo Parquet será gravado.
    """
    print(f"Convertendo {input_path} -> {output_path}")
    con = conectar_duckdb_s3()

    try:
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_csv_auto('{sql_path(input_path)}')
            )
            TO '{sql_path(output_path)}'
            (FORMAT PARQUET, COMPRESSION 'ZSTD');
            """
        )

        print(f"Terminou: {output_path}")
        print("=" * 80)
        return
    except Exception as exc:
        print(
            "DuckDB/httpfs falhou ao acessar o CSV no S3. "
            "Validando o objeto com boto3 antes de tentar o fallback local...",
            flush=True,
        )
        duckdb_error = exc
    finally:
        con.close()

    try:
        validar_objeto_s3(input_path)
    except Exception as boto3_exc:
        raise RuntimeError(
            "Falha ao acessar o CSV no S3 tanto pelo DuckDB quanto pelo boto3. "
            "Verifique endpoint, secrets S3_ACCESS_KEY/S3_SECRET_KEY, permissao "
            f"no objeto e se o caminho existe: {input_path}"
        ) from boto3_exc

    print(
        "Objeto validado com boto3. Tentando conversao por arquivo temporario local...",
        flush=True,
    )
    converter_parquet_local_fallback(input_path, output_path)
    print(
        "Conversao concluida pelo fallback local apos falha do DuckDB/httpfs.",
        flush=True,
    )
    print("=" * 80)
    print(f"Erro original do DuckDB/httpfs: {duckdb_error}", flush=True)


def converter_parquet_local_fallback(input_path: str, output_path: str) -> None:
    """Converte CSV S3 para Parquet S3 usando arquivos temporarios locais."""
    with tempfile.TemporaryDirectory(prefix="nids_csv_to_parquet_") as temp_dir:
        temp_path = Path(temp_dir)
        local_csv = temp_path / Path(input_path).name
        local_parquet = temp_path / Path(output_path).name

        download_file_from_s3(input_path, local_csv)

        con = duckdb.connect()
        try:
            con.execute(
                f"""
                COPY (
                    SELECT *
                    FROM read_csv_auto('{sql_path(str(local_csv))}')
                )
                TO '{sql_path(str(local_parquet))}'
                (FORMAT PARQUET, COMPRESSION 'ZSTD');
                """
            )
        finally:
            con.close()

        upload_file_to_s3(local_parquet, output_path)


def limpar_dataset(
    input_path: str,
    output_path: str,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str = "Label",
) -> str:
    """Cria um arquivo Parquet limpo tratando features numéricas e categóricas.

    A função:
    - lê o dataset bruto em Parquet;
    - converte features numéricas para FLOAT;
    - converte features categóricas para INTEGER;
    - converte Label para TINYINT;
    - remove valores NULL;
    - remove valores infinitos das numéricas;
    - mantém apenas Label 0 e 1;
    - salva o dataset limpo em Parquet com compressão ZSTD.

    Args:
        input_path: Caminho do dataset bruto.
        output_path: Caminho onde o arquivo Parquet limpo será gravado.
        numeric_features: Colunas numéricas usadas no modelo.
        categorical_features: Colunas categóricas/códigos usadas no modelo.
        target: Coluna alvo. Por padrão, "Label".

    Returns:
        Caminho do arquivo Parquet limpo gerado.
    """

    print("Limpando dataset...")
    con = conectar_duckdb_s3()

    try:
        print(f"\nAnalisando dataset ORIGINAL: {input_path}")

        rows_before = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_path(input_path)}')"
        ).fetchone()[0]

        dist_before = con.execute(
            f"""
            SELECT {to_string(target)}, COUNT(*) AS total
            FROM read_parquet('{sql_path(input_path)}')
            GROUP BY {to_string(target)}
            ORDER BY {to_string(target)}
            """
        ).fetchall()

        print(f"Linhas antes: {rows_before:,}")
        print("Distribuição antes:", dist_before)

        numeric_casts = ", ".join(
            f"TRY_CAST({to_string(col)} AS FLOAT) AS {to_string(col)}"
            for col in numeric_features
        )

        categorical_casts = ", ".join(
            f"TRY_CAST({to_string(col)} AS INTEGER) AS {to_string(col)}"
            for col in categorical_features
        )

        all_casts = ", ".join(
            part for part in [numeric_casts, categorical_casts] if part
        )

        all_features = numeric_features + categorical_features

        condicao_null = " AND ".join(
            f"{to_string(col)} IS NOT NULL" for col in all_features + [target]
        )

        condicao_finite = " AND ".join(
            f"isfinite({to_string(col)})" for col in numeric_features
        )

        sql = f"""
        COPY (
            WITH typed AS (
                SELECT
                    {all_casts},
                    TRY_CAST({to_string(target)} AS TINYINT) AS {to_string(target)}
                FROM read_parquet('{sql_path(input_path)}')
            )
            SELECT *
            FROM typed
            WHERE
                {to_string(target)} IN (0, 1)
                AND {condicao_null}
                AND {condicao_finite}
        )
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION 'ZSTD');
        """

        con.execute(sql)

        print(f"\nAnalisando dataset LIMPO: {output_path}")

        rows_after = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_path(output_path)}')"
        ).fetchone()[0]

        dist_after = con.execute(
            f"""
            SELECT {to_string(target)}, COUNT(*) AS total
            FROM read_parquet('{sql_path(output_path)}')
            GROUP BY {to_string(target)}
            ORDER BY {to_string(target)}
            """
        ).fetchall()

        print(f"Linhas depois: {rows_after:,}")
        print("Distribuição depois:", dist_after)

        removidos = rows_before - rows_after
        removidos_perct = (removidos / rows_before) * 100 if rows_before > 0 else 0

        print("\nImpacto da limpeza:")
        print(f"Removidas: {removidos:,} linhas")
        print(f"% removido: {removidos_perct:.2f}%")
        print(f"Arquivo criado: {output_path}")
        print("Dataset limpo.")
        print("=" * 80)

        return output_path

    finally:
        con.close()


def separar_dataset(
    input_path: str,
    output_path: str,
    dataset_name: str,
    hash_features: list[str],
    target: str = "Label",
    train_size: float = 0.60,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> tuple[str, str, str]:
    """Separa o dataset em treino, validação e teste usando DuckDB.

    Args:
        input_path: Caminho do arquivo Parquet limpo de entrada.
        output_path: Caminho/prefixo onde os splits serão salvos.
        dataset_name: Nome usado nos arquivos de saída.
        hash_features: Colunas usadas para gerar o hash determinístico.
        target: Coluna alvo.
        train_size: Proporção destinada ao treino.
        val_size: Proporção destinada à validação.
        random_seed: Seed usada no hash.

    Returns:
        Caminhos dos arquivos de treino, validação e teste.
    """
    con = conectar_duckdb_s3()

    try:
        train_file, val_file, test_file = split_dataset(
            con=con,
            source_file=input_path,
            output_path=output_path,
            dataset_name=dataset_name,
            hash_columns=hash_features,
            target=target,
            train_size=train_size,
            val_size=val_size,
            random_seed=random_seed,
        )

        print(
            f"Dataset separado em:\n"
            f"Train: {train_file}\n"
            f"Val: {val_file}\n"
            f"Test: {test_file}"
        )

        return train_file, val_file, test_file

    finally:
        con.close()


def scaler_dataset(
    con: duckdb.DuckDBPyConnection,
    input_path: str,
    output_path: str,
    numeric_features: list[str],
    categorical_features: list[str],
    stats: dict[str, tuple[float, float]],
    categories: dict[str, list[int]],
    target: str = "Label",
) -> str:
    """Padroniza numéricas, aplica one-hot e mantém o alvo.

    Args:
        con: Conexão DuckDB configurada.
        input_path: Caminho do Parquet de entrada.
        output_path: Caminho onde o Parquet processado será salvo.
        numeric_features: Colunas numéricas a padronizar.
        categorical_features: Colunas categóricas a codificar.
        stats: Estatísticas calculadas no treino.
        categories: Categorias calculadas no treino.
        target: Nome da coluna alvo.

    Returns:
        Caminho do arquivo processado.
    """

    numeric_exprs = []

    for col in numeric_features:
        mean, std = stats[col]

        numeric_exprs.append(
            f"""
            CAST(
                ({to_string(col)} - {mean}) / {std}
                AS FLOAT
            ) AS {to_string(col)}
            """
        )

    categorical_exprs = []

    for col in categorical_features:
        for value in categories[col]:
            safe_col_name = f"{col}_{value}"

            categorical_exprs.append(
                f"""
                CAST(
                    CASE
                        WHEN {to_string(col)} = {value} THEN 1
                        ELSE 0
                    END
                    AS TINYINT
                ) AS {to_string(safe_col_name)}
                """
            )

    select_expr = ", ".join(numeric_exprs + categorical_exprs + [to_string(target)])

    sql = f"""
    COPY (
        SELECT
            {select_expr}
        FROM read_parquet('{sql_path(input_path)}')
    )
    TO '{sql_path(output_path)}'
    (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """

    print(f"Processando para treino: {input_path}")
    con.execute(sql)
    print(f"Arquivo criado: {output_path}")

    return output_path


def preparar_datasets_treino(
    train_path: str,
    val_path: str,
    test_path: str,
    output_path: str,
    dataset_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str = "Label",
) -> tuple[str, str, str]:
    """Prepara os datasets finais para treino da MLP.

    Faz:
    - padronização das numéricas usando estatísticas do treino;
    - one-hot encoding das categóricas usando categorias do treino;
    - mantém Label como alvo.
    """

    con = conectar_duckdb_s3()

    try:
        stats = calcular_estatisticas_treino(
            con=con,
            train_path=train_path,
            numeric_features=numeric_features,
        )

        categories = calcular_categorias_treino(
            con=con,
            train_path=train_path,
            categorical_features=categorical_features,
        )

        print("Categorias encontradas no treino:")
        for col, values in categories.items():
            print(f"{col}: {len(values)} categorias")

        train_processed = f"{output_path}/{dataset_name}_train_processed.parquet"
        val_processed = f"{output_path}/{dataset_name}_val_processed.parquet"
        test_processed = f"{output_path}/{dataset_name}_test_processed.parquet"

        scaler_dataset(
            con=con,
            input_path=train_path,
            output_path=train_processed,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            stats=stats,
            categories=categories,
            target=target,
        )

        scaler_dataset(
            con=con,
            input_path=val_path,
            output_path=val_processed,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            stats=stats,
            categories=categories,
            target=target,
        )

        scaler_dataset(
            con=con,
            input_path=test_path,
            output_path=test_processed,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            stats=stats,
            categories=categories,
            target=target,
        )

        print("Arquivos processados para treino:")
        print(f"Train processed: {train_processed}")
        print(f"Val processed: {val_processed}")
        print(f"Test processed: {test_processed}")

        print(f"Finalizado: {dataset_name}")
        print("=" * 80)

        return train_processed, val_processed, test_processed

    finally:
        con.close()


def listar_colunas_features(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
    target: str = "Label",
) -> list[str]:
    """Lista as colunas de entrada do modelo, removendo a coluna alvo.

    Args:
        con: Conexão DuckDB configurada.
        parquet_path: Caminho do arquivo Parquet.
        target: Nome da coluna alvo.

    Returns:
        Lista de colunas usadas como entrada no PCA.
    """
    rows = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{sql_path(parquet_path)}')
        """
    ).fetchall()

    columns = [row[0] for row in rows]
    feature_columns = [col for col in columns if col != target]

    if not feature_columns:
        raise ValueError(
            f"Nenhuma feature encontrada em {parquet_path}. "
            f"A coluna target usada foi: {target}"
        )

    return feature_columns


def iterar_batches_features(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
    feature_columns: list[str],
    batch_size: int = 100_000,
) -> Iterator[np.ndarray]:
    """Itera sobre batches de features a partir de um Parquet.

    Args:
        con: Conexão DuckDB configurada.
        parquet_path: Caminho do arquivo Parquet.
        feature_columns: Colunas usadas como entrada.
        batch_size: Quantidade aproximada de linhas por batch.

    Yields:
        Array NumPy float32 com as features do batch.
    """
    select_expr = ", ".join(to_string(col) for col in feature_columns)

    cursor = con.execute(
        f"""
        SELECT {select_expr}
        FROM read_parquet('{sql_path(parquet_path)}')
        """
    )

    vectors_per_chunk = max(1, math.ceil(batch_size / 2048))

    while True:
        df = cursor.fetch_df_chunk(vectors_per_chunk=vectors_per_chunk)

        if df is None or df.empty:
            break

        yield df.astype("float32").to_numpy(copy=False)


def ajustar_pca_treino(
    con: duckdb.DuckDBPyConnection,
    train_path: str,
    feature_columns: list[str],
    n_components: int = 30,
    batch_size: int = 100_000,
) -> tuple[IncrementalPCA, float]:
    """Ajusta o PCA usando somente o dataset de treino.

    Args:
        con: Conexão DuckDB configurada.
        train_path: Caminho do Parquet de treino processado.
        feature_columns: Colunas usadas como entrada.
        n_components: Quantidade de componentes principais.
        batch_size: Tamanho aproximado dos batches.

    Returns:
        Tupla contendo o PCA ajustado e a variância explicada.
    """
    if n_components <= 0:
        raise ValueError("n_components precisa ser maior que zero.")

    if n_components > len(feature_columns):
        raise ValueError(
            f"n_components={n_components} é maior que o número de features "
            f"disponíveis: {len(feature_columns)}."
        )

    pca = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size,
    )

    print(
        f"Ajustando PCA com {n_components} componentes usando somente o treino...",
        flush=True,
    )

    total_linhas = 0
    batch_num = 0
    fitted_batches = 0

    for batch in iterar_batches_features(
        con=con,
        parquet_path=train_path,
        feature_columns=feature_columns,
        batch_size=batch_size,
    ):
        batch_num += 1
        linhas_batch = batch.shape[0]
        total_linhas += linhas_batch

        print(
            f"[PCA FIT] Batch {batch_num} | "
            f"linhas no batch: {linhas_batch:,} | "
            f"total processado: {total_linhas:,}",
            flush=True,
        )

        if linhas_batch < n_components:
            print(
                f"[PCA FIT] Batch {batch_num} ignorado: "
                f"{linhas_batch:,} linhas < {n_components} componentes.",
                flush=True,
            )
            continue

        pca.partial_fit(batch)
        fitted_batches += 1

    if fitted_batches == 0:
        raise ValueError(
            "Nenhum batch foi usado para ajustar o PCA. "
            f"Verifique se o treino tem pelo menos {n_components} linhas."
        )

    variancia = float(np.sum(pca.explained_variance_ratio_))

    print(
        f"[PCA FIT] Finalizado | "
        f"batches lidos: {batch_num} | "
        f"batches usados: {fitted_batches} | "
        f"linhas lidas: {total_linhas:,}",
        flush=True,
    )
    print(f"Variância explicada pelo PCA: {variancia:.4f}", flush=True)

    return pca, variancia


def transformar_split_pca(
    con: duckdb.DuckDBPyConnection,
    input_path: str,
    output_path: str,
    pca: IncrementalPCA,
    feature_columns: list[str],
    target: str = "Label",
    batch_size: int = 100_000,
) -> str:
    """Aplica PCA em um split e salva um novo Parquet.

    Args:
        con: Conexão DuckDB configurada.
        input_path: Caminho do Parquet processado de entrada.
        output_path: Caminho do Parquet PCA de saída.
        pca: PCA ajustado no treino.
        feature_columns: Colunas usadas como entrada.
        target: Nome da coluna alvo.
        batch_size: Tamanho aproximado dos batches.

    Returns:
        Caminho do arquivo PCA gerado.
    """
    print(f"Aplicando PCA: {input_path}", flush=True)

    component_columns = [f"PC_{idx + 1:03d}" for idx in range(pca.n_components_)]

    local_tmp = tempfile.NamedTemporaryFile(
        suffix=".parquet",
        delete=False,
    )
    local_tmp_path = local_tmp.name
    local_tmp.close()

    writer: pq.ParquetWriter | None = None

    select_columns = ", ".join(
        [to_string(col) for col in feature_columns] + [to_string(target)]
    )

    cursor = con.execute(
        f"""
        SELECT {select_columns}
        FROM read_parquet('{sql_path(input_path)}')
        """
    )

    vectors_per_chunk = max(1, math.ceil(batch_size / 2048))

    batch_num = 0
    total_linhas = 0

    try:
        while True:
            df = cursor.fetch_df_chunk(vectors_per_chunk=vectors_per_chunk)

            if df is None or df.empty:
                break

            batch_num += 1
            linhas_batch = len(df)
            total_linhas += linhas_batch

            print(
                f"[PCA TRANSFORM] {input_path} | "
                f"batch {batch_num} | "
                f"linhas no batch: {linhas_batch:,} | "
                f"total processado: {total_linhas:,}",
                flush=True,
            )

            x = df[feature_columns].astype("float32").to_numpy(copy=False)
            y = df[target].astype("int8").to_numpy(copy=False)

            x_pca = pca.transform(x).astype("float32")

            df_pca = pd.DataFrame(
                x_pca,
                columns=component_columns,
            )
            df_pca[target] = y

            table = pa.Table.from_pandas(df_pca, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    local_tmp_path,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    if batch_num == 0:
        Path(local_tmp_path).unlink(missing_ok=True)
        raise ValueError(f"Nenhuma linha encontrada para aplicar PCA: {input_path}")

    print(
        f"[PCA TRANSFORM] Gravando resultado no destino: {output_path}",
        flush=True,
    )

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet('{sql_path(local_tmp_path)}')
        )
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION 'ZSTD');
        """
    )

    Path(local_tmp_path).unlink(missing_ok=True)

    print(
        f"[PCA TRANSFORM] Finalizado: {input_path} | "
        f"total de batches: {batch_num} | "
        f"total de linhas: {total_linhas:,}",
        flush=True,
    )
    print(f"Arquivo PCA criado: {output_path}", flush=True)

    return output_path


def salvar_artefatos_pca_no_bucket(
    pca: IncrementalPCA,
    feature_columns: list[str],
    artifact_path: str,
    dataset_name: str,
    explained_variance: float,
    target: str = "Label",
) -> dict[str, str]:
    """Salva os artefatos do PCA localmente e envia para o bucket.

    Args:
        pca: PCA ajustado no treino.
        feature_columns: Colunas usadas como entrada do PCA.
        artifact_path: Prefixo S3 onde os artefatos serão salvos.
        dataset_name: Nome do dataset.
        explained_variance: Variância explicada total.
        target: Nome da coluna alvo.

    Returns:
        Dicionário com as URIs S3 dos artefatos enviados.
    """
    local_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_name}_pca_artifacts_"))

    try:
        pca_file = local_dir / f"{dataset_name}_pca.joblib"
        features_file = local_dir / f"{dataset_name}_pca_feature_columns.json"
        metadata_file = local_dir / f"{dataset_name}_pca_metadata.json"

        joblib.dump(pca, pca_file)

        with features_file.open("w", encoding="utf-8") as file:
            json.dump(
                feature_columns,
                file,
                ensure_ascii=False,
                indent=2,
            )

        metadata = {
            "dataset_name": dataset_name,
            "target": target,
            "n_components": int(pca.n_components_),
            "n_features_in": int(pca.n_features_in_),
            "explained_variance": explained_variance,
            "explained_variance_ratio": [
                float(value) for value in pca.explained_variance_ratio_
            ],
            "component_columns": [
                f"PC_{idx + 1:03d}" for idx in range(pca.n_components_)
            ],
        }

        with metadata_file.open("w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        pca_s3 = join_s3_path(artifact_path, pca_file.name)
        features_s3 = join_s3_path(artifact_path, features_file.name)
        metadata_s3 = join_s3_path(artifact_path, metadata_file.name)

        upload_file_to_s3(pca_file, pca_s3)
        upload_file_to_s3(features_file, features_s3)
        upload_file_to_s3(metadata_file, metadata_s3)

        print("Artefatos PCA enviados ao bucket:", flush=True)
        print(f"PCA: {pca_s3}", flush=True)
        print(f"Features: {features_s3}", flush=True)
        print(f"Metadata: {metadata_s3}", flush=True)

        return {
            "pca": pca_s3,
            "feature_columns": features_s3,
            "metadata": metadata_s3,
        }

    finally:
        shutil.rmtree(local_dir, ignore_errors=True)


def aplicar_pca_datasets(
    train_path: str,
    val_path: str,
    test_path: str,
    output_path: str,
    dataset_name: str,
    artifact_path: str,
    target: str = "Label",
    n_components: int = 30,
    batch_size: int = 100_000,
) -> tuple[str, str, str]:
    """Aplica PCA nos datasets processados de treino, validação e teste.

    O PCA é ajustado somente no treino e depois aplicado em treino,
    validação e teste.

    Args:
        train_path: Caminho do treino processado.
        val_path: Caminho da validação processada.
        test_path: Caminho do teste processado.
        output_path: Caminho/prefixo onde os arquivos PCA serão salvos.
        dataset_name: Nome do dataset usado nos arquivos de saída.
        artifact_path: Caminho onde os artefatos serão salvos.
        target: Nome da coluna alvo.
        n_components: Quantidade de componentes principais.
        batch_size: Tamanho aproximado dos batches.

    Returns:
        Caminhos dos arquivos train, val e test após PCA.
    """
    con = conectar_duckdb_s3()

    try:
        feature_columns = listar_colunas_features(
            con=con,
            parquet_path=train_path,
            target=target,
        )

        print(f"Features antes do PCA: {len(feature_columns)}", flush=True)

        pca, explained_variance = ajustar_pca_treino(
            con=con,
            train_path=train_path,
            feature_columns=feature_columns,
            n_components=n_components,
            batch_size=batch_size,
        )

        train_pca = join_s3_path(output_path, f"{dataset_name}_train_pca.parquet")
        val_pca = join_s3_path(output_path, f"{dataset_name}_val_pca.parquet")
        test_pca = join_s3_path(output_path, f"{dataset_name}_test_pca.parquet")

        transformar_split_pca(
            con=con,
            input_path=train_path,
            output_path=train_pca,
            pca=pca,
            feature_columns=feature_columns,
            target=target,
            batch_size=batch_size,
        )

        transformar_split_pca(
            con=con,
            input_path=val_path,
            output_path=val_pca,
            pca=pca,
            feature_columns=feature_columns,
            target=target,
            batch_size=batch_size,
        )

        transformar_split_pca(
            con=con,
            input_path=test_path,
            output_path=test_pca,
            pca=pca,
            feature_columns=feature_columns,
            target=target,
            batch_size=batch_size,
        )

        salvar_artefatos_pca_no_bucket(
            pca=pca,
            feature_columns=feature_columns,
            artifact_path=artifact_path,
            dataset_name=dataset_name,
            explained_variance=explained_variance,
            target=target,
        )

        print("Arquivos PCA gerados:", flush=True)
        print(f"Train PCA: {train_pca}", flush=True)
        print(f"Val PCA: {val_pca}", flush=True)
        print(f"Test PCA: {test_pca}", flush=True)
        print("=" * 80, flush=True)

        return train_pca, val_pca, test_pca

    finally:
        con.close()

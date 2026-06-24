"""Utilitários para DuckDB e S3."""

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import duckdb
from botocore.config import Config

from .config import S3_ENDPOINT, S3_URL_STYLE, load_s3_credentials


def conf_s3(con: duckdb.DuckDBPyConnection) -> None:
    """Configura uma conexão DuckDB para acessar o bucket S3 do projeto.

    Args:
        con: Conexão DuckDB que receberá extensão, endpoint e credenciais.
    """
    access_key, secret_key = load_s3_credentials()

    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET s3_endpoint=?;", [S3_ENDPOINT])
    con.execute("SET s3_url_style=?;", [S3_URL_STYLE])
    con.execute("SET s3_access_key_id=?;", [access_key])
    con.execute("SET s3_secret_access_key=?;", [secret_key])


def conectar_duckdb_s3() -> duckdb.DuckDBPyConnection:
    """Cria uma conexão DuckDB já configurada para acessar o S3.

    Returns:
        Conexão DuckDB pronta para ler e gravar nos buckets do projeto.
    """
    con = duckdb.connect()
    conf_s3(con)
    return con


def to_string(col: str) -> str:
    """Monta o identificador SQL de uma coluna preservando o nome original.

    Args:
        col: Nome da coluna.

    Returns:
        Nome da coluna entre aspas duplas para uso em SQL.
    """
    return f'"{col}"'


def sql_path(path: str) -> str:
    """Escapa aspas simples em paths interpolados em comandos SQL.

    Args:
        path: Caminho usado dentro de uma string SQL.

    Returns:
        Caminho com aspas simples escapadas.
    """
    return str(path).replace("'", "''")


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Separa uma URI S3 em bucket e chave.

    Args:
        s3_uri: URI no formato s3://bucket/caminho/arquivo.

    Returns:
        Tupla contendo o nome do bucket e a chave do objeto.
    """
    parsed = urlparse(s3_uri)

    if parsed.scheme != "s3":
        raise ValueError(
            f"URI inválida: {s3_uri}. Use o formato s3://bucket/caminho/arquivo."
        )

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket or not key:
        raise ValueError(
            f"URI S3 incompleta: {s3_uri}. Use o formato s3://bucket/caminho/arquivo."
        )

    return bucket, key


def endpoint_url() -> str:
    """Monta a URL do endpoint S3.

    Returns:
        URL completa do endpoint S3.
    """
    if S3_ENDPOINT.startswith("http://") or S3_ENDPOINT.startswith("https://"):
        return S3_ENDPOINT

    return f"https://{S3_ENDPOINT}"


def criar_cliente_s3(
    aws_region: str | None = None,
) -> Any:
    """Cria cliente S3 compatível com AWS/Contabo.

    Usa a configuração compartilhada do projeto:
    - S3_ENDPOINT;
    - S3_URL_STYLE;
    - load_s3_credentials().

    Args:
        aws_region: Região AWS/S3. Pode ser None para provedores compatíveis.

    Returns:
        Cliente boto3 S3.
    """
    access_key, secret_key = load_s3_credentials()

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url(),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
        config=Config(
            s3={
                "addressing_style": S3_URL_STYLE,
            }
        ),
    )


def upload_file_to_s3(
    local_path: str | Path,
    s3_uri: str,
    aws_region: str | None = None,
) -> None:
    """Envia um arquivo local para o bucket S3.

    Args:
        local_path: Caminho local do arquivo.
        s3_uri: URI de destino no S3.
        aws_region: Região AWS/S3. Pode ser None para Contabo.
    """
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {local_path}")

    if not local_path.is_file():
        raise ValueError(f"O caminho informado não é um arquivo: {local_path}")

    bucket, key = parse_s3_uri(s3_uri)
    client = criar_cliente_s3(aws_region=aws_region)

    print(f"Enviando artefato: {local_path} -> {s3_uri}", flush=True)

    client.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
    )

    print(f"Artefato enviado: {s3_uri}", flush=True)


def validar_objeto_s3(s3_uri: str, aws_region: str | None = None) -> None:
    """Valida se um objeto S3 existe e esta acessivel via boto3."""
    bucket, key = parse_s3_uri(s3_uri)
    client = criar_cliente_s3(aws_region=aws_region)
    client.head_object(Bucket=bucket, Key=key)


def download_file_from_s3(
    s3_uri: str,
    local_path: str | Path,
    aws_region: str | None = None,
) -> None:
    """Baixa um objeto S3 para um arquivo local."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    bucket, key = parse_s3_uri(s3_uri)
    client = criar_cliente_s3(aws_region=aws_region)
    print(f"Baixando {s3_uri} -> {local_path}", flush=True)
    client.download_file(Bucket=bucket, Key=key, Filename=str(local_path))


def join_s3_path(base_path: str, *parts: str) -> str:
    """Junta caminhos S3 evitando barras duplicadas.

    Args:
        base_path: Caminho base, por exemplo `s3://bucket/prefixo`.
        parts: Partes adicionais do caminho.

    Returns:
        Caminho completo com uma única barra entre cada parte.
    """
    return "/".join([base_path.rstrip("/"), *(str(part).strip("/") for part in parts)])

"""Configurações compartilhadas dos notebooks NIDS."""

SEED = 42

NORMAL_LABEL = 0

ATTACK_LABEL = 1

CUSTOM_CALLBACK = "val_macro_f1"

S3_ENDPOINT = "usc1.contabostorage.com"

S3_URL_STYLE = "path"

RAW_BUCKET = "s3://bronze/nids-dataset"

SILVER_BUCKET = "s3://silver/nids-dataset"

RAW_BOT_CSV = f"{RAW_BUCKET}/NF-BoT-IoT-v3/data/NF-BoT-IoT-v3.csv"

RAW_CICIDS_CSV = f"{RAW_BUCKET}/NF-CICIDS2018-v3/data/NF-CICIDS2018-v3.csv"

RAW_TON_CSV = f"{RAW_BUCKET}/NF-ToN-IoT-v3/data/NF-ToN-IoT-v3.csv"

RAW_UNSW_CSV = f"{RAW_BUCKET}/NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.csv"

RAW_BOT_PARQUET = f"{RAW_BUCKET}/NF-BoT-IoT-v3/data/NF-BoT-IoT-v3.parquet"

RAW_CICIDS_PARQUET = f"{RAW_BUCKET}/NF-CICIDS2018-v3/data/NF-CICIDS2018-v3.parquet"

RAW_TON_PARQUET = f"{RAW_BUCKET}/NF-ToN-IoT-v3/data/NF-ToN-IoT-v3.parquet"

RAW_UNSW_PARQUET = f"{RAW_BUCKET}/NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.parquet"

CLEAN_BOT_PARQUET = f"{SILVER_BUCKET}/NF-BoT-IoT-v3/data/NF-BoT-IoT-v3_clean.parquet"

CLEAN_CICIDS_PARQUET = (
    f"{SILVER_BUCKET}/NF-CICIDS2018-v3/data/NF-CICIDS2018-v3_clean.parquet"
)

CLEAN_TON_PARQUET = f"{SILVER_BUCKET}/NF-ToN-IoT-v3/data/NF-ToN-IoT-v3_clean.parquet"

CLEAN_UNSW_PARQUET = (
    f"{SILVER_BUCKET}/NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3_clean.parquet"
)

SPLIT_BOT = f"{SILVER_BUCKET}/NF-BoT-IoT-v3/data"

SPLIT_CICIDS = f"{SILVER_BUCKET}/NF-CICIDS2018-v3/data"

SPLIT_TON = f"{SILVER_BUCKET}/NF-ToN-IoT-v3/data"

SPLIT_UNSW = f"{SILVER_BUCKET}/NF-UNSW-NB15-v3/data"

BOT = {
    "RAW_CSV": RAW_BOT_CSV,
    "RAW_PARQUET": RAW_BOT_PARQUET,
    "CLEAN_PARQUET": CLEAN_BOT_PARQUET,
    "SPLIT_PATH": SPLIT_BOT,
    "DATASET_KEY": "NF-BoT-IoT-v3",
    "ARTIFACT_PATH": f"{SILVER_BUCKET}/NF-BoT-IoT-v3/artifacts",
}

CICIDS = {
    "RAW_CSV": RAW_CICIDS_CSV,
    "RAW_PARQUET": RAW_CICIDS_PARQUET,
    "CLEAN_PARQUET": CLEAN_CICIDS_PARQUET,
    "SPLIT_PATH": SPLIT_CICIDS,
    "DATASET_KEY": "NF-CICIDS2018-v3",
    "ARTIFACT_PATH": f"{SILVER_BUCKET}/NF-CICIDS2018-v3/artifacts",
}

TON = {
    "RAW_CSV": RAW_TON_CSV,
    "RAW_PARQUET": RAW_TON_PARQUET,
    "CLEAN_PARQUET": CLEAN_TON_PARQUET,
    "SPLIT_PATH": SPLIT_TON,
    "DATASET_KEY": "NF-ToN-IoT-v3",
    "ARTIFACT_PATH": f"{SILVER_BUCKET}/NF-ToN-IoT-v3/artifacts",
}

UNSW = {
    "RAW_CSV": RAW_UNSW_CSV,
    "RAW_PARQUET": RAW_UNSW_PARQUET,
    "CLEAN_PARQUET": CLEAN_UNSW_PARQUET,
    "SPLIT_PATH": SPLIT_UNSW,
    "DATASET_KEY": "NF-UNSW-NB15-v3",
    "ARTIFACT_PATH": f"{SILVER_BUCKET}/NF-UNSW-NB15-v3/artifacts",
}

NUMERIC_FEATURES = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

TARGET = "Label"

CATEGORICAL_FEATURES = [
    "PROTOCOL",
    "L7_PROTO",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
]

HASH_FEATURES = [
    "Label",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]

FEATURES_DROP = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "Attack",
]

def load_s3_credentials() -> tuple[str, str]:
    """Carrega as credenciais S3 salvas no Secrets do Google Colab.

    Returns:
        Tupla com access key e secret key do S3.

    Raises:
        RuntimeError: Quando a função é executada fora do Colab ou quando uma
            das credenciais obrigatórias não foi definida.
    """
    try:
        from google.colab import userdata
    except ImportError as exc:
        raise RuntimeError(
            "load_s3_credentials depende do Google Colab. "
            "Execute no Colab ou adapte esta função para variáveis de ambiente."
        ) from exc

    access_key = userdata.get("S3_ACCESS_KEY")
    secret_key = userdata.get("S3_SECRET_KEY")

    if not access_key or not secret_key:
        raise RuntimeError("S3_ACCESS_KEY e S3_SECRET_KEY faltando")

    return access_key, secret_key

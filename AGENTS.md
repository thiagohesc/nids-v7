# Instruções para o Codex

## Padrão de código Python

- Use type hints em todas as funções e métodos.
- Toda função deve ter tipo de retorno explícito.
- Use `None` como retorno explícito para funções que não retornam valor.
- Use Google Style docstrings.
- Não use NumPy Style nem reStructuredText Style.
- Não deixe funções públicas sem docstring.
- Não documente o óbvio; explique objetivo, parâmetros, retorno, efeitos colaterais e decisões técnicas importantes.
- Quando corrigir código existente, preserve a lógica original sempre que possível.
- Ao encontrar funções sem tipagem, adicione tipagem completa nos parâmetros e no retorno.
- Ao encontrar funções sem docstring, adicione docstring em Google Style.
- Prefira tipos modernos do Python:
  - `list[str]` em vez de `List[str]`
  - `dict[str, int]` em vez de `Dict[str, int]`
  - `str | None` em vez de `Optional[str]`
- Para funções que recebem conexão DuckDB, use `duckdb.DuckDBPyConnection` quando o pacote `duckdb` estiver importado.
- Para funções que recebem DataFrame, use `pd.DataFrame`.
- Para funções que recebem Series, use `pd.Series`.
- Para arrays NumPy, use `np.ndarray`.

## Padrão de docstring

Use este formato:

```python
def nome_funcao(parametro: Tipo) -> TipoRetorno:
    """Resumo curto da função.

    Explicação adicional quando necessário.

    Args:
        parametro: Descrição objetiva do parâmetro.

    Returns:
        Descrição objetiva do retorno.

    Raises:
        TipoErro: Quando a exceção pode acontecer.
    """
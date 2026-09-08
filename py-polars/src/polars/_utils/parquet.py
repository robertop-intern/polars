from collections.abc import Callable, Mapping, Sequence
from typing import Any

from polars._typing import (
    ParquetColumnStatistics,
    ParquetMetadataContext,
    ParquetMetadataFn,
    ParquetStatistics,
)

_DEFAULT_STATISTICS: dict[str, bool] = {
    "min": True,
    "max": True,
    "distinct_count": False,
    "null_count": True,
}
_FULL_STATISTICS: dict[str, bool] = dict.fromkeys(_DEFAULT_STATISTICS, True)


def _normalize_column_statistics(
    statistics: ParquetColumnStatistics,
) -> dict[str, bool]:
    """
    Normalize parquet statistic options for a single column.

    The options are normalized to a dictionary of False/True entries.
    """
    if isinstance(statistics, bool):
        return dict(_DEFAULT_STATISTICS) if statistics else {}
    elif isinstance(statistics, Mapping):
        return dict(statistics)
    elif statistics == "full":
        return dict(_FULL_STATISTICS)
    else:
        msg = f"invalid value for `statistics`: {statistics!r}"
        raise ValueError(msg)


def normalize_statistics(
    statistics: ParquetStatistics,
) -> dict[str, bool] | list[tuple[str, dict[str, bool]]]:
    """
    Normalize ParquetStatistics for easier processing in Rust.

    statistics are either made into a dictionary of options
    and these options are applied to ALL parquet columns
    or into a list of tuples where the string defines the column
    and the dictionary the options for that parquet column
    """
    if isinstance(statistics, (bool, str, Mapping)):
        return _normalize_column_statistics(statistics)

    if not isinstance(statistics, Sequence):
        msg = f"invalid value for `statistics`: {statistics!r}"
        raise TypeError(msg)

    per_column: list[tuple[str, dict[str, bool]]] = []
    seen: set[str] = set()
    name: str
    spec: ParquetColumnStatistics
    for entry in statistics:
        if isinstance(entry, str):
            name, spec = entry, True
        elif isinstance(entry, tuple):
            name, spec = entry
        else:
            msg = (
                "per-column `statistics` entries must be a column name or a"
                f" (column name, statistics) tuple, got {entry!r}"
            )
            raise TypeError(msg)

        if name in seen:
            msg = f"column {name!r} occurs more than once in `statistics`"
            raise ValueError(msg)
        seen.add(name)

        normalize_spec = _normalize_column_statistics(spec)
        if normalize_spec:
            per_column.append((name, normalize_spec))

    return per_column


def wrap_parquet_metadata_callback(
    fn: ParquetMetadataFn,
) -> Callable[[Any], list[tuple[str, str]]]:
    def pyo3_compatible_callback(ctx: Any) -> list[tuple[str, str]]:
        ctx_py = ParquetMetadataContext(
            arrow_schema=ctx.arrow_schema,
        )
        return list(fn(ctx_py).items())

    return pyo3_compatible_callback

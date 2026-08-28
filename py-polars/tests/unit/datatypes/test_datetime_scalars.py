from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

import polars as pl
from polars.exceptions import (
    ComputeError,
    InvalidOperationError,
    OutOfBoundsError,
    PanicException,
    SchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Callable


NS_MIN = datetime(1677, 9, 21, 0, 12, 43, 145225)  # minimal us value fitting in ns
NS_MAX = datetime(2262, 4, 11, 23, 47, 16, 854775)  # max us value fitting in ns
NS_UNDERFLOW = datetime(1677, 9, 21, 0, 12, 43, 145224)  # one us below the ns range

# `NS_UNDERFLOW` is -9223372036854776000 ns which is below i64::MIN and should underflow to:
NS_UNDERFLOW_WRAPPED = datetime(2262, 4, 11, 23, 47, 16, 854775)

# An unremarkable value, used wherever a test needs a second datetime.
ANCHOR = datetime(2000, 1, 1, 12, 30, 15, 123456)
SRC = pl.Series([ANCHOR])

NS = pl.Datetime("ns")

OVERFLOW_ERRORS = (
    ComputeError,
    InvalidOperationError,
    OutOfBoundsError,
    OverflowError,
    PanicException,
    SchemaError,
)


@dataclass(frozen=True)
class DtypeOp:
    """An operation that can be told which precision to produce."""

    name: str
    create: Callable[[datetime, pl.DataType], Any]
    extract: Callable[[Any], datetime | None] = field(default=lambda result: result[0])


class DatetimeNamedTuple(NamedTuple):
    """Named-tuple row containing a datetime."""

    t: datetime


@dataclass
class DatetimeDataclass:
    """Dataclass row containing a datetime."""

    t: datetime


DTYPE_OPS = [
    # Series constructors
    DtypeOp(
        "series_list",
        lambda value, dtype: pl.Series([value], dtype=dtype),
    ),  # Series.__init__ -> sequence_to_pyseries -> new_from_any_values
    DtypeOp(
        "series_tuple",
        lambda value, dtype: pl.Series((value,), dtype=dtype),
    ),  # tuple sequence -> sequence_to_pyseries
    DtypeOp(
        "series_generator",
        lambda value, dtype: pl.Series((item for item in [value]), dtype=dtype),
    ),  # generator is materialized before sequence_to_pyseries
    DtypeOp(
        "series_null_leading",
        lambda value, dtype: pl.Series([None, value], dtype=dtype),
        lambda result: result[1],
    ),  # first-non-null dtype inference
    DtypeOp(
        "series_nested_list",
        lambda value, dtype: pl.Series([[value]], dtype=pl.List(dtype)),
        lambda result: result[0][0],
    ),  # nested List AnyValue builder
    DtypeOp(
        "series_nested_array",
        lambda value, dtype: pl.Series([[value]], dtype=pl.Array(dtype, 1)),
        lambda result: result[0][0],
    ),  # nested Array builder
    DtypeOp(
        "series_nested_struct",
        lambda value, dtype: pl.Series([{"t": value}], dtype=pl.Struct({"t": dtype})),
        lambda result: result[0]["t"],
    ),  # Struct field row builder
    # Series functions with a dtype/return_dtype argument
    DtypeOp(
        "series_replace_strict",
        lambda value, dtype: SRC.replace_strict(ANCHOR, value, return_dtype=dtype),
    ),  # pyo3 callback result -> from_any_values_and_dtype
    # Expr functions with a dtype argument
    DtypeOp(
        "expr_replace_strict",
        lambda value, dtype: pl.lit(ANCHOR).replace_strict(
            ANCHOR, value, return_dtype=dtype
        ),
        lambda result: pl.select(result.alias("t")).get_column("t")[0],
    ),
    DtypeOp(
        "lit",
        lambda value, dtype: pl.lit(value, dtype=dtype),
        lambda result: pl.select(result.alias("t")).get_column("t")[0],
    ),
    DtypeOp(
        "repeat_eager",
        lambda value, dtype: pl.repeat(value, 1, dtype=dtype, eager=True),
    ),
    DtypeOp(
        "repeat_lazy",
        lambda value, dtype: pl.repeat(value, 1, dtype=dtype),
        lambda result: pl.select(result.alias("t")).get_column("t")[0],
    ),
    DtypeOp(
        "datetime_range_eager",
        lambda value, dtype: pl.datetime_range(
            value, value, time_unit=dtype.time_unit, eager=True
        ),
    ),
    DtypeOp(
        "datetime_range_lazy",
        lambda value, dtype: pl.datetime_range(value, value, time_unit=dtype.time_unit),
        lambda result: pl.select(result.alias("t")).get_column("t")[0],
    ),
    DtypeOp(
        "datetime_ranges_eager",
        lambda value, dtype: pl.datetime_ranges(
            value, value, time_unit=dtype.time_unit, eager=True
        ),
        lambda result: result[0][0],
    ),
    DtypeOp(
        "datetime_ranges_lazy",
        lambda value, dtype: pl.datetime_ranges(
            value, value, time_unit=dtype.time_unit
        ),
        lambda result: pl.select(result.alias("t")).get_column("t")[0][0],
    ),
    # DataFrame constructors
    DtypeOp(
        "dataframe_scalar_dict",
        lambda value, dtype: pl.DataFrame({"t": value}, schema={"t": dtype}),
        lambda result: result.get_column("t")[0],
    ),  # scalar-only dict -> Series
    DtypeOp(
        "dataframe_scalar_broadcast",
        lambda value, dtype: pl.DataFrame(
            {"x": [1, 2], "t": value}, schema_overrides={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # scalar beside vector -> repeat(dtype=..., eager=True)
    DtypeOp(
        "dataframe_column_sequence",
        lambda value, dtype: pl.DataFrame({"t": [value]}, schema={"t": dtype}),
        lambda result: result.get_column("t")[0],
    ),  # dict sequence -> Series(values, dtype=schema dtype)
    DtypeOp(
        "dataframe_column_sequence_overrides",
        lambda value, dtype: pl.DataFrame(
            {"t": [value]}, schema_overrides={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # same Series path, target supplied by schema_overrides
    DtypeOp(
        "dataframe_scalar_sequence",
        lambda value, dtype: pl.DataFrame([value], schema={"t": dtype}),
        lambda result: result.get_column("t")[0],
    ),  # sequence-of-elements dispatcher -> one Series column
    DtypeOp(
        "dataframe_generator_column",
        lambda value, dtype: pl.DataFrame(
            {"t": (item for item in [value])}, schema={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # generator column is materialized before Series construction
    DtypeOp(
        "dataframe_rows_list",
        lambda value, dtype: pl.DataFrame([[value]], schema={"t": dtype}, orient="row"),
        lambda result: result.get_column("t")[0],
    ),  # sequence_to_pydf -> PyDataFrame.from_rows -> AnyValueBuffer
    DtypeOp(
        "dataframe_rows_schema_overrides",
        lambda value, dtype: pl.DataFrame(
            [[value]], schema=["t"], schema_overrides={"t": dtype}, orient="row"
        ),
        lambda result: result.get_column("t")[0],
    ),  # from_rows target merged from schema_overrides
    DtypeOp(
        "dataframe_rows_tuple",
        lambda value, dtype: pl.DataFrame(
            [(value,)], schema={"t": dtype}, orient="row"
        ),
        lambda result: result.get_column("t")[0],
    ),  # tuple row -> same from_rows sink
    DtypeOp(
        "dataframe_rows_dict",
        lambda value, dtype: pl.DataFrame([{"t": value}], schema={"t": dtype}),
        lambda result: result.get_column("t")[0],
    ),  # mapping record -> PyDataFrame.from_dicts
    DtypeOp(
        "dataframe_dict_rows_schema_overrides",
        lambda value, dtype: pl.DataFrame(
            [{"t": value}], schema_overrides={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # from_dicts target supplied by schema_overrides
    DtypeOp(
        "dataframe_rows_namedtuple",
        lambda value, dtype: pl.DataFrame(
            [DatetimeNamedTuple(value)], schema={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # namedtuple dispatcher -> from_rows
    DtypeOp(
        "dataframe_rows_dataclass",
        lambda value, dtype: pl.DataFrame(
            [DatetimeDataclass(value)], schema={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # dataclass dispatcher -> from_dicts
    DtypeOp(
        "dataframe_generator_rows",
        lambda value, dtype: pl.DataFrame(
            ([item] for item in [value]), schema={"t": dtype}, orient="row"
        ),
        lambda result: result.get_column("t")[0],
    ),  # iterable_to_pydf -> row buffer
    DtypeOp(
        "dataframe_generator_dicts",
        lambda value, dtype: pl.DataFrame(
            ({"t": item} for item in [value]), schema={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # iterable_to_pydf -> mapping buffer
    DtypeOp(
        "dataframe_scalar_struct",
        lambda value, dtype: pl.DataFrame(
            {"s": {"t": value}}, schema_overrides={"s": pl.Struct({"t": dtype})}
        ),
        lambda result: result.get_column("s").struct.field("t")[0],
    ),  # explicit Struct override -> nested from_dicts row builder
    # Top-level from_* constructors
    DtypeOp(
        "from_dict_schema",
        lambda value, dtype: pl.from_dict({"t": [value]}, schema={"t": dtype}),
        lambda result: result.get_column("t")[0],
    ),  # from_dict -> dict_to_pydf
    DtypeOp(
        "from_dict_schema_overrides",
        lambda value, dtype: pl.from_dict(
            {"t": [value]}, schema_overrides={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # schema_overrides -> dict sequence Series
    DtypeOp(
        "from_dicts_schema",
        lambda value, dtype: pl.from_dicts([{"t": value}], schema={"t": dtype}),
        lambda result: result.get_column("t")[0],
    ),  # from_dicts -> DataFrame.from_dicts
    DtypeOp(
        "from_dicts_schema_overrides",
        lambda value, dtype: pl.from_dicts(
            [{"t": value}], schema_overrides={"t": dtype}
        ),
        lambda result: result.get_column("t")[0],
    ),  # inferred schema plus explicit target override
    DtypeOp(
        "from_records_rows",
        lambda value, dtype: pl.from_records(
            [[value]], schema={"t": dtype}, orient="row"
        ),
        lambda result: result.get_column("t")[0],
    ),  # from_records -> from_rows
    DtypeOp(
        "from_records_schema_overrides",
        lambda value, dtype: pl.from_records(
            [[value]], schema=["t"], schema_overrides={"t": dtype}, orient="row"
        ),
        lambda result: result.get_column("t")[0],
    ),  # from_records -> from_rows with merged schema override
    DtypeOp(
        "from_records_columns",
        lambda value, dtype: pl.from_records(
            [[value]], schema={"t": dtype}, orient="col"
        ),
        lambda result: result.get_column("t")[0],
    ),  # column-oriented records take the Series constructor, not the row builder
    # Post-cast operations
    DtypeOp(
        "postcast_series_list",
        lambda value, dtype: pl.Series([value]).cast(dtype),
    ),
    DtypeOp(
        "postcast_series_replace",
        lambda value, dtype: SRC.replace(ANCHOR, value).cast(dtype),
    ),
    DtypeOp(
        "postcast_lit",
        lambda value, dtype: pl.lit(value).cast(dtype),
        lambda result: pl.select(result.alias("t")).get_column("t")[0],
    ),
    DtypeOp(
        "postcast_dataframe_rows",
        lambda value, dtype: (
            pl.DataFrame([[value]], schema=["t"], orient="row")
            .get_column("t")
            .cast(dtype)
        ),
    ),
]


@pytest.mark.parametrize("op", DTYPE_OPS, ids=lambda op: op.name)
def test_datetime_cast_to_ns(op: DtypeOp) -> None:
    # Casting an underflowing us datetime to ns should throw while underflowing values don't
    for value in (NS_MIN, NS_MAX):
        assert op.extract(op.create(value, NS)) == value

    try:
        result = op.extract(op.create(NS_UNDERFLOW, NS))
    except OVERFLOW_ERRORS:
        return

    assert result == NS_UNDERFLOW_WRAPPED, f"expected a raise; got {result!r}"
    pytest.fail(f"underflow wrapped around i64::MIN to {NS_UNDERFLOW_WRAPPED}")


NON_STRICT_OPS = [
    DtypeOp(
        "series_replace",
        lambda value, dtype: SRC.replace(ANCHOR, value, return_dtype=dtype),
    ),
    DtypeOp(
        "expr_replace",
        lambda value, dtype: pl.lit(ANCHOR).replace(ANCHOR, value, return_dtype=dtype),
        lambda result: pl.select(result.alias("t")).get_column("t")[0],
    ),
    DtypeOp(
        "series_list",
        lambda value, dtype: pl.Series([value], dtype=dtype, strict=False),
    ),
    DtypeOp(
        "series_tuple",
        lambda value, dtype: pl.Series((value,), dtype=dtype, strict=False),
    ),
    DtypeOp(
        "series_generator",
        lambda value, dtype: pl.Series(
            (item for item in [value]), dtype=dtype, strict=False
        ),
    ),
    DtypeOp(
        "series_null_leading",
        lambda value, dtype: pl.Series([None, value], dtype=dtype, strict=False),
        lambda result: result[1],
    ),
    DtypeOp(
        "series_nested_list",
        lambda value, dtype: pl.Series([[value]], dtype=pl.List(dtype), strict=False),
        lambda result: result[0][0],
    ),
    DtypeOp(
        "series_nested_struct",
        lambda value, dtype: pl.Series(
            [{"t": value}], dtype=pl.Struct({"t": dtype}), strict=False
        ),
        lambda result: result[0]["t"],
    ),
    DtypeOp(
        "dataframe_column_sequence",
        lambda value, dtype: pl.DataFrame(
            {"t": [value]}, schema={"t": dtype}, strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "dataframe_rows_list",
        lambda value, dtype: pl.DataFrame(
            [[value]], schema={"t": dtype}, orient="row", strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "dataframe_rows_tuple",
        lambda value, dtype: pl.DataFrame(
            [(value,)], schema={"t": dtype}, orient="row", strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "dataframe_rows_dict",
        lambda value, dtype: pl.DataFrame(
            [{"t": value}], schema={"t": dtype}, strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "dataframe_rows_namedtuple",
        lambda value, dtype: pl.DataFrame(
            [DatetimeNamedTuple(value)], schema={"t": dtype}, strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "dataframe_rows_dataclass",
        lambda value, dtype: pl.DataFrame(
            [DatetimeDataclass(value)], schema={"t": dtype}, strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "dataframe_generator_dicts",
        lambda value, dtype: pl.DataFrame(
            ({"t": item} for item in [value]), schema={"t": dtype}, strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "from_dict",
        lambda value, dtype: pl.from_dict(
            {"t": [value]}, schema={"t": dtype}, strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "from_dicts",
        lambda value, dtype: pl.from_dicts(
            [{"t": value}], schema={"t": dtype}, strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "from_records_rows",
        lambda value, dtype: pl.from_records(
            [[value]], schema={"t": dtype}, orient="row", strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "from_records_columns",
        lambda value, dtype: pl.from_records(
            [[value]], schema={"t": dtype}, orient="col", strict=False
        ),
        lambda result: result.get_column("t")[0],
    ),
    DtypeOp(
        "postcast_series_list",
        lambda value, dtype: pl.Series([value]).cast(dtype, strict=False),
    ),
]


@pytest.mark.filterwarnings(
    "ignore:the `return_dtype` parameter for `replace` is deprecated"
)
@pytest.mark.parametrize("op", NON_STRICT_OPS, ids=lambda op: op.name)
def test_datetime_cast_to_ns_non_strict(op: DtypeOp) -> None:
    # a non strict cast of an underflowing value is None

    assert op.extract(op.create(NS_UNDERFLOW, NS)) is None

use std::sync::Arc;

use arrow::datatypes::{ArrowSchema, ArrowSchemaRef};
use polars_core::prelude::CompatLevel;
use polars_error::{PolarsResult, polars_ensure};
use polars_parquet::write::{
    BrotliLevel, CompressionOptions, GzipLevel, StatisticsOptions, Version, WriteOptions, ZstdLevel,
};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::KeyValueMetadata;

#[derive(Default, Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ParquetWriteOptions {
    /// Data page compression
    pub compression: ParquetCompression,
    /// Compute and write column statistics.
    pub statistics: ParquetStatisticsConfig,
    /// If `None` will be all written to a single row group.
    pub row_group_size: Option<usize>,
    /// if `None` will be 1024^2 bytes
    pub data_page_size: Option<usize>,
    /// Custom file-level key value metadata
    pub key_value_metadata: Option<KeyValueMetadata>,
    pub arrow_schema: Option<ArrowSchemaRef>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub compat_level: Option<CompatLevel>,
}

impl ParquetWriteOptions {
    pub fn compat_level(&self) -> CompatLevel {
        self.compat_level.unwrap_or(CompatLevel::oldest())
    }
}

/// Which statistics to compute and write, for every column or per column.
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ParquetStatisticsConfig {
    /// The same statistics for every column.
    Global(StatisticsOptions),
    /// Statistics for named columns only. Columns that are not named get none.
    PerColumn(Vec<(PlSmallStr, StatisticsOptions)>),
}

impl Default for ParquetStatisticsConfig {
    fn default() -> Self {
        Self::Global(StatisticsOptions::default())
    }
}

impl From<StatisticsOptions> for ParquetStatisticsConfig {
    fn from(statistics: StatisticsOptions) -> Self {
        Self::Global(statistics)
    }
}

impl ParquetStatisticsConfig {
    /// The [`StatisticsOptions`] to use for each field of `schema`
    ///
    /// Returns a [`StatisticsOptions`] per column in the schema, in order of the columns
    pub fn resolve(&self, schema: &ArrowSchema) -> PolarsResult<Vec<StatisticsOptions>> {
        let per_column = match self {
            Self::Global(statistics) => return Ok(vec![*statistics; schema.len()]),
            Self::PerColumn(per_column) => per_column,
        };

        let mut out = vec![StatisticsOptions::empty(); schema.len()];
        let mut is_set = vec![false; schema.len()];
        for (name, statistics) in per_column {
            let idx = schema.try_index_of(name)?;
            polars_ensure!(
                !std::mem::replace(&mut is_set[idx], true),
                Duplicate: "column {:?} occurs more than once in `statistics`", name.as_str(),
            );
            out[idx] = *statistics;
        }
        Ok(out)
    }
}

/// Schema-resolved options shared by the Parquet column encoders.
///
/// Statistics are resolved against the write schema on construction, so the encoders can select
/// the options of a column by its index with [`Self::for_column`].
#[derive(Clone, Debug)]
pub struct FullWriteOptions {
    compression: CompressionOptions,
    version: Version,
    data_page_size: Option<usize>,
    column_statistics: Arc<[StatisticsOptions]>,
}

impl FullWriteOptions {
    pub fn try_new(
        compression: CompressionOptions,
        version: Version,
        data_page_size: Option<usize>,
        statistics: &ParquetStatisticsConfig,
        schema: &ArrowSchema,
    ) -> PolarsResult<Self> {
        Ok(Self {
            compression,
            version,
            data_page_size,
            column_statistics: statistics.resolve(schema)?.into(),
        })
    }

    /// The options used to initialize the file writer.
    ///
    /// The statistics are the union over all columns, as they only decide whether the file
    /// writes statistics at all. What a column writes is selected with [`Self::for_column`].
    pub fn file_options(&self) -> WriteOptions {
        let statistics =
            self.column_statistics
                .iter()
                .fold(StatisticsOptions::empty(), |acc, statistics| {
                    StatisticsOptions {
                        min_value: acc.min_value || statistics.min_value,
                        max_value: acc.max_value || statistics.max_value,
                        distinct_count: acc.distinct_count || statistics.distinct_count,
                        null_count: acc.null_count || statistics.null_count,
                        binary_statistics_truncate_length: acc
                            .binary_statistics_truncate_length
                            .or(statistics.binary_statistics_truncate_length),
                    }
                });

        self.options_with_statistics(statistics)
    }

    /// The [WriteOptions] options for a column index
    pub fn for_column(&self, index: usize) -> WriteOptions {
        self.options_with_statistics(self.column_statistics[index])
    }

    fn options_with_statistics(&self, statistics: StatisticsOptions) -> WriteOptions {
        WriteOptions {
            statistics,
            compression: self.compression,
            version: self.version,
            data_page_size: self.data_page_size,
        }
    }
}

/// The compression strategy to use for writing Parquet files.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ParquetCompression {
    Uncompressed,
    Snappy,
    Gzip(Option<GzipLevel>),
    Brotli(Option<BrotliLevel>),
    Zstd(Option<ZstdLevel>),
    Lz4Raw,
}

impl Default for ParquetCompression {
    fn default() -> Self {
        Self::Zstd(None)
    }
}

impl From<ParquetCompression> for CompressionOptions {
    fn from(value: ParquetCompression) -> Self {
        use ParquetCompression::*;
        match value {
            Uncompressed => CompressionOptions::Uncompressed,
            Snappy => CompressionOptions::Snappy,
            Gzip(level) => CompressionOptions::Gzip(level),
            Brotli(level) => CompressionOptions::Brotli(level),
            Lz4Raw => CompressionOptions::Lz4Raw,
            Zstd(level) => CompressionOptions::Zstd(level),
        }
    }
}

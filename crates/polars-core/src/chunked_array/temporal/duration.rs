use chrono::Duration as ChronoDuration;

use crate::chunked_array::cast::CastOptions;
use crate::fmt::{fmt_duration_string, iso_duration_string};
use crate::prelude::DataType::Duration;
use crate::prelude::*;
use crate::utils::handle_casting_failures;

impl DurationChunked {
    pub fn time_unit(&self) -> TimeUnit {
        match &self.dtype {
            DataType::Duration(tu) => *tu,
            _ => unreachable!(),
        }
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    ///
    /// Raises if a value does not fit in `tu`.
    pub fn cast_time_unit(&self, tu: TimeUnit) -> PolarsResult<Self> {
        self.cast_time_unit_with_options(tu, CastOptions::Strict)
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    pub fn cast_time_unit_with_options(
        &self,
        tu: TimeUnit,
        options: CastOptions,
    ) -> PolarsResult<Self> {
        let current_unit = self.time_unit();
        let mut out = self.clone();
        out.set_time_unit(tu);

        use crate::datatypes::time_unit::TimeUnit::*;
        let (divisor, multiplier) = match (current_unit, tu) {
            (Nanoseconds, Microseconds) => (Some(1_000i64), None),
            (Nanoseconds, Milliseconds) => (Some(1_000_000i64), None),
            (Microseconds, Milliseconds) => (Some(1_000i64), None),
            (Microseconds, Nanoseconds) => (None, Some(1_000i64)),
            (Milliseconds, Microseconds) => (None, Some(1_000i64)),
            (Milliseconds, Nanoseconds) => (None, Some(1_000_000i64)),
            (Nanoseconds, Nanoseconds)
            | (Microseconds, Microseconds)
            | (Milliseconds, Milliseconds) => return Ok(out),
        };

        match (divisor, multiplier) {
            (None, None) | (Some(_), Some(_)) => unreachable!(),
            (Some(divisor), _) => out.phys = (&self.phys).wrapping_trunc_div_scalar(divisor),
            (_, Some(multiplier)) => match options {
                CastOptions::Overflowing => {
                    out.phys = self.phys.as_ref().wrapping_mul_scalar(multiplier)
                },
                CastOptions::NonStrict => {
                    out.phys = self.phys.as_ref().checked_mul_scalar(multiplier)
                },
                CastOptions::Strict => {
                    out.phys = self.phys.as_ref().checked_mul_scalar(multiplier);
                    if out.phys.null_count() != self.phys.null_count() {
                        handle_casting_failures(
                            &self.clone().into_series(),
                            &out.clone().into_series(),
                        )?;
                    }
                },
            },
        }
        Ok(out)
    }

    /// Change the underlying [`TimeUnit`]. This does not modify the data.
    pub fn set_time_unit(&mut self, tu: TimeUnit) {
        self.dtype = Duration(tu);
    }

    /// Convert from [`Duration`] to String; note that `strftime` format
    /// strings are not supported, only the specifiers 'iso' and 'polars'.
    pub fn to_string(&self, format: &str) -> PolarsResult<StringChunked> {
        // the duration string functions below can reuse this string buffer
        let mut s = String::with_capacity(32);
        match format {
            "iso" | "iso:strict" => {
                let out: StringChunked =
                    self.phys
                        .apply_nonnull_values_generic(DataType::String, |v: i64| {
                            s.clear();
                            iso_duration_string(&mut s, v, self.time_unit());
                            s.clone()
                        });
                Ok(out)
            },
            "polars" => {
                let out: StringChunked =
                    self.phys
                        .apply_nonnull_values_generic(DataType::String, |v: i64| {
                            s.clear();
                            fmt_duration_string(&mut s, v, self.time_unit())
                                .map_err(|e| polars_err!(ComputeError: "{:?}", e))
                                .expect("failed to format duration");
                            s.clone()
                        });
                Ok(out)
            },
            _ => {
                polars_bail!(
                    InvalidOperation: "format {:?} not supported for Duration type (expected one of 'iso' or 'polars')",
                    format
                )
            },
        }
    }

    /// Construct a new [`DurationChunked`] from an iterator over [`ChronoDuration`].
    pub fn from_duration<I: IntoIterator<Item = ChronoDuration>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Microseconds => |v: ChronoDuration| v.num_microseconds().unwrap(),
            TimeUnit::Milliseconds => |v: ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.into_iter().map(func).collect::<Vec<_>>();
        Int64Chunked::from_vec(name, vals).into_duration(tu)
    }

    /// Construct a new [`DurationChunked`] from an iterator over optional [`ChronoDuration`].
    pub fn from_duration_options<I: IntoIterator<Item = Option<ChronoDuration>>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Microseconds => |v: ChronoDuration| v.num_microseconds().unwrap(),
            TimeUnit::Milliseconds => |v: ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.into_iter().map(|opt| opt.map(func));
        Int64Chunked::from_iter_options(name, vals).into_duration(tu)
    }
}

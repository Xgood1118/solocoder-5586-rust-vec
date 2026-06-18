use serde::{Deserialize, Serialize};
use num_traits::{Zero, One, Float};
use crate::vector::{Vector, FloatScalar};
use crate::error::{VecMathError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBin {
    pub start: FloatScalar,
    pub end: FloatScalar,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStatistics {
    pub count: usize,
    pub mean: FloatScalar,
    pub variance: FloatScalar,
    pub std_dev: FloatScalar,
    pub min: FloatScalar,
    pub max: FloatScalar,
    pub quantiles: Vec<(f64, FloatScalar)>,
    pub histogram: Vec<HistogramBin>,
    pub nan_count: usize,
    pub inf_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    pub per_dimension: Vec<VectorStatistics>,
}

pub fn compute_statistics(
    values: &[FloatScalar],
    quantiles: &[f64],
    bin_count: usize,
) -> Result<VectorStatistics> {
    let mut clean_values: Vec<FloatScalar> = Vec::new();
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;

    for &v in values {
        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        } else {
            clean_values.push(v);
        }
    }

    if clean_values.is_empty() {
        return Err(VecMathError::InsufficientData { expected: 1, got: 0 });
    }

    let n = clean_values.len() as FloatScalar;
    let zero = FloatScalar::zero();

    let mut sum = zero;
    let mut min_val = clean_values[0];
    let mut max_val = clean_values[0];

    for &v in &clean_values {
        sum += v;
        if v < min_val { min_val = v; }
        if v > max_val { max_val = v; }
    }

    let mean = sum / n;

    let mut var_sum = zero;
    for &v in &clean_values {
        let diff = v - mean;
        var_sum += diff * diff;
    }
    let variance = if clean_values.len() > 1 {
        var_sum / (n - FloatScalar::one())
    } else {
        zero
    };
    let std_dev = variance.sqrt();

    clean_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut q_results = Vec::new();
    let n = clean_values.len();
    for &q in quantiles {
        if !(0.0..=1.0).contains(&q) {
            return Err(VecMathError::InvalidParameter {
                param: "quantile".to_string(),
                value: q.to_string(),
                reason: "must be in [0, 1]".to_string(),
            });
        }
        let pos = q * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        if lower == upper {
            q_results.push((q, clean_values[lower]));
        } else {
            let frac: FloatScalar = num_traits::cast(pos - pos.floor()).unwrap();
            let one = FloatScalar::one();
            let val = clean_values[lower] * (one - frac) + clean_values[upper] * frac;
            q_results.push((q, val));
        }
    }

    let histogram = if bin_count > 0 {
        compute_histogram(&clean_values, min_val, max_val, bin_count)
    } else {
        Vec::new()
    };

    Ok(VectorStatistics {
        count: clean_values.len(),
        mean,
        variance,
        std_dev,
        min: min_val,
        max: max_val,
        quantiles: q_results,
        histogram,
        nan_count,
        inf_count,
    })
}

fn compute_histogram(
    sorted_values: &[FloatScalar],
    min: FloatScalar,
    max: FloatScalar,
    bin_count: usize,
) -> Vec<HistogramBin> {
    let mut bins = Vec::with_capacity(bin_count);
    let range = if max > min { max - min } else { FloatScalar::epsilon() };
    let bin_width = range / num_traits::cast::<usize, FloatScalar>(bin_count).unwrap();

    for i in 0..bin_count {
        let start = min + num_traits::cast::<usize, FloatScalar>(i).unwrap() * bin_width;
        let end = start + bin_width;
        bins.push(HistogramBin { start, end, count: 0 });
    }

    for &v in sorted_values {
        let mut bin_idx = ((v - min) / bin_width).floor() as usize;
        if bin_idx >= bin_count { bin_idx = bin_count - 1; }
        if bin_idx < bin_count {
            bins[bin_idx].count += 1;
        }
    }

    bins
}

pub fn compute_batch_statistics(
    vectors: &[Vector],
    quantiles: &[f64],
    bin_count: usize,
) -> Result<BatchStatistics> {
    if vectors.is_empty() {
        return Err(VecMathError::InsufficientData { expected: 1, got: 0 });
    }

    let dim = vectors[0].dim();
    for v in vectors {
        v.check_dimension(dim)?;
    }

    let mut per_dimension = Vec::with_capacity(dim);
    for d in 0..dim {
        let values: Vec<FloatScalar> = vectors.iter().map(|v| v[d]).collect();
        per_dimension.push(compute_statistics(&values, quantiles, bin_count)?);
    }

    Ok(BatchStatistics { per_dimension })
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stats() {
        let values: Vec<FloatScalar> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_statistics(&values, &[0.5], 2).unwrap();
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nan_inf_filter() {
        let values: Vec<FloatScalar> = vec![1.0, FloatScalar::NAN, 2.0, FloatScalar::INFINITY, 3.0];
        let stats = compute_statistics(&values, &[], 0).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 1);
    }

    #[test]
    fn test_batch_stats() {
        let vectors = vec![
            Vector::new(vec![1.0, 10.0]),
            Vector::new(vec![2.0, 20.0]),
            Vector::new(vec![3.0, 30.0]),
        ];
        let batch = compute_batch_statistics(&vectors, &[], 0).unwrap();
        assert_eq!(batch.per_dimension.len(), 2);
        assert!((batch.per_dimension[0].mean - 2.0).abs() < 1e-10);
        assert!((batch.per_dimension[1].mean - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_interpolation() {
        let values: Vec<FloatScalar> = (1..=10).map(|i| num_traits::cast(i).unwrap()).collect();
        let stats = compute_statistics(&values, &[0.25, 0.5, 0.75, 0.95, 0.99], 0).unwrap();
        assert_eq!(stats.quantiles.len(), 5);

        let q25 = stats.quantiles.iter().find(|(q, _)| *q == 0.25).map(|(_, v)| *v).unwrap();
        let q50 = stats.quantiles.iter().find(|(q, _)| *q == 0.50).map(|(_, v)| *v).unwrap();
        let q75 = stats.quantiles.iter().find(|(q, _)| *q == 0.75).map(|(_, v)| *v).unwrap();

        assert!((q25 - num_traits::cast::<f64, FloatScalar>(3.25).unwrap()).abs() < num_traits::cast::<f64, FloatScalar>(1e-10).unwrap(), "p25 expected 3.25, got {}", q25);
        assert!((q50 - num_traits::cast::<f64, FloatScalar>(5.5).unwrap()).abs() < num_traits::cast::<f64, FloatScalar>(1e-10).unwrap(), "p50 expected 5.5, got {}", q50);
        assert!((q75 - num_traits::cast::<f64, FloatScalar>(7.75).unwrap()).abs() < num_traits::cast::<f64, FloatScalar>(1e-10).unwrap(), "p75 expected 7.75, got {}", q75);
    }
}

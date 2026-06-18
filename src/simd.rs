use crate::vector::FloatScalar;

pub fn dot_product(a: &[FloatScalar], b: &[FloatScalar]) -> FloatScalar {
    #[cfg(all(target_arch = "x86_64", feature = "f64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_avx2_f64(a, b) };
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "f32"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_avx2_f32(a, b) };
        }
    }
    dot_product_scalar(a, b)
}

fn dot_product_scalar(a: &[FloatScalar], b: &[FloatScalar]) -> FloatScalar {
    let n = a.len().min(b.len());
    let mut sum = FloatScalar::zero();
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

pub fn squared_l2_norm(a: &[FloatScalar]) -> FloatScalar {
    #[cfg(all(target_arch = "x86_64", feature = "f64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { squared_l2_norm_avx2_f64(a) };
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "f32"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { squared_l2_norm_avx2_f32(a) };
        }
    }
    squared_l2_norm_scalar(a)
}

fn squared_l2_norm_scalar(a: &[FloatScalar]) -> FloatScalar {
    let mut sum = FloatScalar::zero();
    for &v in a {
        sum += v * v;
    }
    sum
}

pub fn l1_norm(a: &[FloatScalar]) -> FloatScalar {
    let mut sum = FloatScalar::zero();
    for &v in a {
        sum += v.abs();
    }
    sum
}

pub fn linf_norm(a: &[FloatScalar]) -> FloatScalar {
    let mut max_val = FloatScalar::zero();
    for &v in a {
        let abs_v = v.abs();
        if abs_v > max_val {
            max_val = abs_v;
        }
    }
    max_val
}

pub fn euclidean_distance_squared(a: &[FloatScalar], b: &[FloatScalar]) -> FloatScalar {
    let n = a.len().min(b.len());
    let mut sum = FloatScalar::zero();
    for i in 0..n {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

pub fn manhattan_distance(a: &[FloatScalar], b: &[FloatScalar]) -> FloatScalar {
    let n = a.len().min(b.len());
    let mut sum = FloatScalar::zero();
    for i in 0..n {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

pub fn chebyshev_distance(a: &[FloatScalar], b: &[FloatScalar]) -> FloatScalar {
    let n = a.len().min(b.len());
    let mut max_val = FloatScalar::zero();
    for i in 0..n {
        let diff = (a[i] - b[i]).abs();
        if diff > max_val {
            max_val = diff;
        }
    }
    max_val
}

#[cfg(all(target_arch = "x86_64", feature = "f64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2_f64(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut i = 0;
    let mut sum = 0.0f64;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(va, vb);
        sum += _mm256_cvtsd_f64(_mm256_hadd_pd(prod, prod));
        let upper = _mm256_extractf128_pd::<1>(prod);
        sum += _mm_cvtsd_f64(_mm_hadd_pd(upper, upper));
        i += 4;
    }
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(all(target_arch = "x86_64", feature = "f32"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut i = 0;
    let mut sum = 0.0f32;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(va, vb);
        let hadd = _mm256_hadd_ps(prod, prod);
        let hadd2 = _mm256_hadd_ps(hadd, hadd);
        let lo = _mm256_castps256_ps128(hadd2);
        let hi = _mm256_extractf128_ps::<1>(hadd2);
        sum += _mm_cvtss_f32(lo) + _mm_cvtss_f32(hi);
        i += 8;
    }
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(all(target_arch = "x86_64", feature = "f64"))]
#[target_feature(enable = "avx2")]
unsafe fn squared_l2_norm_avx2_f64(a: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    let mut sum = 0.0f64;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let prod = _mm256_mul_pd(va, va);
        sum += _mm256_cvtsd_f64(_mm256_hadd_pd(prod, prod));
        let upper = _mm256_extractf128_pd::<1>(prod);
        sum += _mm_cvtsd_f64(_mm_hadd_pd(upper, upper));
        i += 4;
    }
    while i < n {
        sum += a[i] * a[i];
        i += 1;
    }
    sum
}

#[cfg(all(target_arch = "x86_64", feature = "f32"))]
#[target_feature(enable = "avx2")]
unsafe fn squared_l2_norm_avx2_f32(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    let mut sum = 0.0f32;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let prod = _mm256_mul_ps(va, va);
        let hadd = _mm256_hadd_ps(prod, prod);
        let hadd2 = _mm256_hadd_ps(hadd, hadd);
        let lo = _mm256_castps256_ps128(hadd2);
        let hi = _mm256_extractf128_ps::<1>(hadd2);
        sum += _mm_cvtss_f32(lo) + _mm_cvtss_f32(hi);
        i += 8;
    }
    while i < n {
        sum += a[i] * a[i];
        i += 1;
    }
    sum
}

use num_traits::Zero;

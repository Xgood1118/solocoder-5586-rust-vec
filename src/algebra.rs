use num_traits::{Zero, One};
use crate::vector::{Vector, FloatScalar};
use crate::error::{VecMathError, Result};
use crate::simd;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormType {
    L1,
    L2,
    LInf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceType {
    Euclidean,
    Manhattan,
    Chebyshev,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AngleUnit {
    Radians,
    Degrees,
}

use serde::{Deserialize, Serialize};

pub fn dot(a: &Vector, b: &Vector) -> Result<FloatScalar> {
    a.check_same_dimension(b)?;
    Ok(simd::dot_product(a.data(), b.data()))
}

pub fn cross(a: &Vector, b: &Vector) -> Result<Vector> {
    if a.dim() != 3 {
        return Err(VecMathError::InvalidCrossDimension { dim: a.dim() });
    }
    if b.dim() != 3 {
        return Err(VecMathError::InvalidCrossDimension { dim: b.dim() });
    }
    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let bx = b[0];
    let by = b[1];
    let bz = b[2];
    Ok(Vector::new(vec![
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]))
}

pub fn norm(v: &Vector, norm_type: NormType) -> FloatScalar {
    match norm_type {
        NormType::L1 => simd::l1_norm(v.data()),
        NormType::L2 => simd::squared_l2_norm(v.data()).sqrt(),
        NormType::LInf => simd::linf_norm(v.data()),
    }
}

pub fn normalize(v: &Vector) -> Result<Vector> {
    let n = norm(v, NormType::L2);
    if n == FloatScalar::zero() {
        return Err(VecMathError::ZeroVector);
    }
    Ok(v.mul_scalar(FloatScalar::one() / n))
}

pub fn distance(a: &Vector, b: &Vector, dist_type: DistanceType) -> Result<FloatScalar> {
    a.check_same_dimension(b)?;
    match dist_type {
        DistanceType::Euclidean => Ok(simd::euclidean_distance_squared(a.data(), b.data()).sqrt()),
        DistanceType::Manhattan => Ok(simd::manhattan_distance(a.data(), b.data())),
        DistanceType::Chebyshev => Ok(simd::chebyshev_distance(a.data(), b.data())),
    }
}

pub fn cosine_similarity(a: &Vector, b: &Vector) -> Result<FloatScalar> {
    a.check_same_dimension(b)?;
    let na = norm(a, NormType::L2);
    let nb = norm(b, NormType::L2);
    if na == FloatScalar::zero() || nb == FloatScalar::zero() {
        return Err(VecMathError::ZeroVector);
    }
    let dot = simd::dot_product(a.data(), b.data());
    Ok(dot / (na * nb))
}

pub fn angle(a: &Vector, b: &Vector, unit: AngleUnit) -> Result<FloatScalar> {
    let cos_sim = cosine_similarity(a, b)?;
    let eps: FloatScalar = num_traits::cast(1e-12).unwrap();
    let one = FloatScalar::one();
    let cos_clamped = if cos_sim > one - eps {
        one
    } else if cos_sim < -one + eps {
        -one
    } else {
        cos_sim
    };
    let rad = cos_clamped.acos();
    match unit {
        AngleUnit::Radians => Ok(rad),
        AngleUnit::Degrees => Ok(rad.to_degrees()),
    }
}

pub fn project(u: &Vector, v: &Vector) -> Result<Vector> {
    u.check_same_dimension(v)?;
    let v_sq = simd::squared_l2_norm(v.data());
    if v_sq == FloatScalar::zero() {
        return Err(VecMathError::ZeroVector);
    }
    let dot = simd::dot_product(u.data(), v.data());
    let scale = dot / v_sq;
    Ok(v.mul_scalar(scale))
}

pub fn reject(u: &Vector, v: &Vector) -> Result<Vector> {
    let proj = project(u, v)?;
    u.sub(&proj)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PairwiseResult<T: Serialize> {
    pub index_a: usize,
    pub index_b: usize,
    pub result: T,
}

pub fn pairwise_dot(vectors: &[Vector]) -> Result<Vec<PairwiseResult<FloatScalar>>> {
    if vectors.len() < 2 {
        return Err(VecMathError::InsufficientVectors { expected: 2, got: vectors.len() });
    }
    let mut results = Vec::new();
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            let result = dot(&vectors[i], &vectors[j])?;
            results.push(PairwiseResult { index_a: i, index_b: j, result });
        }
    }
    Ok(results)
}

pub fn pairwise_distance(vectors: &[Vector], dist_type: DistanceType) -> Result<Vec<PairwiseResult<FloatScalar>>> {
    if vectors.len() < 2 {
        return Err(VecMathError::InsufficientVectors { expected: 2, got: vectors.len() });
    }
    let mut results = Vec::new();
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            let result = distance(&vectors[i], &vectors[j], dist_type)?;
            results.push(PairwiseResult { index_a: i, index_b: j, result });
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(dot(&a, &b).unwrap(), 32.0);
    }

    #[test]
    fn test_cross() {
        let a = Vector::new(vec![1.0, 0.0, 0.0]);
        let b = Vector::new(vec![0.0, 1.0, 0.0]);
        let c = cross(&a, &b).unwrap();
        assert_eq!(c, Vector::new(vec![0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_normalize_zero() {
        let z = Vector::zeros(3);
        assert!(normalize(&z).is_err());
    }

    #[test]
    fn test_normalize() {
        let v = Vector::new(vec![3.0, 4.0]);
        let n = normalize(&v).unwrap();
        let l2 = norm(&n, NormType::L2);
        assert!((l2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Vector::new(vec![1.0, 0.0]);
        let b = Vector::new(vec![0.0, 1.0]);
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_project() {
        let u = Vector::new(vec![3.0, 4.0]);
        let v = Vector::new(vec![1.0, 0.0]);
        let p = project(&u, &v).unwrap();
        assert_eq!(p, Vector::new(vec![3.0, 0.0]));
    }
}

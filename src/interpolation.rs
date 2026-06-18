use serde::{Deserialize, Serialize};
use num_traits::{Zero, One, Float};
use crate::vector::{Vector, FloatScalar};
use crate::error::{VecMathError, Result};
use crate::simd;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BezierType {
    Linear,
    Quadratic,
    Cubic,
    Spherical,
}

fn check_t(t: FloatScalar) -> Result<FloatScalar> {
    let zero = FloatScalar::zero();
    let one = FloatScalar::one();
    if t < zero || t > one {
        return Err(VecMathError::InvalidInterpolationParameter { t: format!("{:?}", t) });
    }
    Ok(t)
}

fn lerp(a: &Vector, b: &Vector, t: FloatScalar) -> Result<Vector> {
    a.check_same_dimension(b)?;
    let one = FloatScalar::one();
    let result: Vec<FloatScalar> = a.data()
        .iter()
        .zip(b.data().iter())
        .map(|(av, bv)| (one - t) * av + t * bv)
        .collect();
    Ok(Vector::new(result))
}

pub fn bezier_linear(p0: &Vector, p1: &Vector, t: FloatScalar) -> Result<Vector> {
    let t = check_t(t)?;
    lerp(p0, p1, t)
}

pub fn bezier_quadratic(p0: &Vector, p1: &Vector, p2: &Vector, t: FloatScalar) -> Result<Vector> {
    let t = check_t(t)?;
    p0.check_same_dimension(p1)?;
    p1.check_same_dimension(p2)?;
    let one = FloatScalar::one();
    let two: FloatScalar = num_traits::cast(2.0).unwrap();
    let mt = one - t;
    let mt2 = mt * mt;
    let t2 = t * t;
    let result: Vec<FloatScalar> = (0..p0.dim())
        .map(|i| mt2 * p0[i] + two * mt * t * p1[i] + t2 * p2[i])
        .collect();
    Ok(Vector::new(result))
}

pub fn bezier_cubic(p0: &Vector, p1: &Vector, p2: &Vector, p3: &Vector, t: FloatScalar) -> Result<Vector> {
    let t = check_t(t)?;
    p0.check_same_dimension(p1)?;
    p1.check_same_dimension(p2)?;
    p2.check_same_dimension(p3)?;
    let one = FloatScalar::one();
    let three: FloatScalar = num_traits::cast(3.0).unwrap();
    let mt = one - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;
    let t2 = t * t;
    let t3 = t2 * t;
    let result: Vec<FloatScalar> = (0..p0.dim())
        .map(|i| {
            mt3 * p0[i]
                + three * mt2 * t * p1[i]
                + three * mt * t2 * p2[i]
                + t3 * p3[i]
        })
        .collect();
    Ok(Vector::new(result))
}

fn slerp(p0: &Vector, p1: &Vector, t: FloatScalar) -> Result<Vector> {
    let zero = FloatScalar::zero();
    let one = FloatScalar::one();
    let two: FloatScalar = num_traits::cast(2.0).unwrap();

    let norm0 = simd::squared_l2_norm(p0.data()).sqrt();
    let norm1 = simd::squared_l2_norm(p1.data()).sqrt();

    if norm0 == zero || norm1 == zero {
        return lerp(p0, p1, t);
    }

    let u0: Vec<FloatScalar> = p0.data().iter().map(|&v| v / norm0).collect();
    let u1: Vec<FloatScalar> = p1.data().iter().map(|&v| v / norm1).collect();

    let dot_raw = simd::dot_product(&u0, &u1);
    let one_m_eps: FloatScalar = num_traits::cast(1.0 - 1e-12).unwrap();
    let neg_one_m_eps: FloatScalar = num_traits::cast(-1.0 + 1e-12).unwrap();
    let dot_clamped = dot_raw.max(neg_one_m_eps).min(one_m_eps);

    let omega = dot_clamped.acos();

    if omega < FloatScalar::epsilon() * two {
        return lerp(p0, p1, t);
    }

    let sin_omega = omega.sin();
    let s0 = ((one - t) * omega).sin() / sin_omega;
    let s1 = (t * omega).sin() / sin_omega;

    let result: Vec<FloatScalar> = (0..p0.dim())
        .map(|i| s0 * p0[i] + s1 * p1[i])
        .collect();
    Ok(Vector::new(result))
}

pub fn bezier_spherical(p0: &Vector, p1: &Vector, t: FloatScalar) -> Result<Vector> {
    let t = check_t(t)?;
    p0.check_same_dimension(p1)?;
    slerp(p0, p1, t)
}

pub fn sample_bezier(
    points: &[Vector],
    bezier_type: BezierType,
    samples: usize,
) -> Result<Vec<Vector>> {
    if samples == 0 {
        return Err(VecMathError::InvalidParameter {
            param: "samples".to_string(),
            value: samples.to_string(),
            reason: "must be positive".to_string(),
        });
    }

    let (required, error_expected) = match bezier_type {
        BezierType::Linear => (2, 2usize),
        BezierType::Quadratic => (3, 3usize),
        BezierType::Cubic => (4, 4usize),
        BezierType::Spherical => (2, 2usize),
    };

    if points.len() < required {
        return Err(VecMathError::InsufficientControlPoints {
            expected: error_expected,
            got: points.len(),
        });
    }

    let mut result = Vec::with_capacity(samples);
    for i in 0..samples {
        let t: FloatScalar = num_traits::cast::<f64, FloatScalar>(
            if samples == 1 { 0.0 } else { i as f64 / (samples - 1) as f64 }
        ).unwrap();

        let sample = match bezier_type {
            BezierType::Linear => bezier_linear(&points[0], &points[1], t)?,
            BezierType::Quadratic => bezier_quadratic(&points[0], &points[1], &points[2], t)?,
            BezierType::Cubic => bezier_cubic(&points[0], &points[1], &points[2], &points[3], t)?,
            BezierType::Spherical => bezier_spherical(&points[0], &points[1], t)?,
        };
        result.push(sample);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_bezier() {
        let p0 = Vector::new(vec![0.0, 0.0]);
        let p1 = Vector::new(vec![1.0, 1.0]);
        let mid = bezier_linear(&p0, &p1, 0.5).unwrap();
        assert!((mid[0] - 0.5).abs() < 1e-10);
        assert!((mid[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_t() {
        let p0 = Vector::new(vec![0.0, 0.0]);
        let p1 = Vector::new(vec![1.0, 1.0]);
        assert!(bezier_linear(&p0, &p1, num_traits::cast(1.5).unwrap()).is_err());
        assert!(bezier_linear(&p0, &p1, num_traits::cast(-0.1).unwrap()).is_err());
    }

    #[test]
    fn test_sample_bezier() {
        let p0 = Vector::new(vec![0.0]);
        let p1 = Vector::new(vec![1.0]);
        let samples = sample_bezier(&[p0, p1], BezierType::Linear, 3).unwrap();
        assert_eq!(samples.len(), 3);
        assert!((samples[1][0] - 0.5).abs() < 1e-10);
    }
}

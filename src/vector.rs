use serde::{Deserialize, Serialize};
use num_traits::{Zero, One};
use std::ops::{Add, Sub, Mul, Div, Index, IndexMut};
use std::fmt;
use crate::error::{VecMathError, Result};

#[cfg(feature = "f32")]
pub type FloatScalar = f32;
#[cfg(not(feature = "f32"))]
pub type FloatScalar = f64;

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Vector {
    data: Vec<FloatScalar>,
}

impl Vector {
    pub fn new(data: Vec<FloatScalar>) -> Self {
        Vector { data }
    }

    pub fn zeros(dim: usize) -> Self {
        Vector { data: vec![FloatScalar::zero(); dim] }
    }

    pub fn ones(dim: usize) -> Self {
        Vector { data: vec![FloatScalar::one(); dim] }
    }

    pub fn from_scalar(scalar: FloatScalar, dim: usize) -> Self {
        Vector { data: vec![scalar; dim] }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[FloatScalar] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [FloatScalar] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<FloatScalar> {
        self.data
    }

    pub fn validate(&self) -> Result<()> {
        if self.data.is_empty() {
            return Err(VecMathError::EmptyVector);
        }
        for (i, &v) in self.data.iter().enumerate() {
            if v.is_nan() {
                return Err(VecMathError::NaNInInput { index: i });
            }
            if v.is_infinite() {
                return Err(VecMathError::InfInInput { index: i });
            }
        }
        Ok(())
    }

    pub fn check_dimension(&self, expected: usize) -> Result<()> {
        if self.dim() != expected {
            return Err(VecMathError::DimensionMismatch {
                expected,
                got: self.dim(),
            });
        }
        Ok(())
    }

    pub fn check_same_dimension(&self, other: &Vector) -> Result<()> {
        if self.dim() != other.dim() {
            return Err(VecMathError::DimensionMismatch {
                expected: self.dim(),
                got: other.dim(),
            });
        }
        Ok(())
    }

    pub fn add(&self, other: &Vector) -> Result<Vector> {
        self.check_same_dimension(other)?;
        let result: Vec<FloatScalar> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Vector::new(result))
    }

    pub fn sub(&self, other: &Vector) -> Result<Vector> {
        self.check_same_dimension(other)?;
        let result: Vec<FloatScalar> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Vector::new(result))
    }

    pub fn mul_elementwise(&self, other: &Vector) -> Result<Vector> {
        self.check_same_dimension(other)?;
        let result: Vec<FloatScalar> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Vector::new(result))
    }

    pub fn div_elementwise(&self, other: &Vector) -> Result<Vector> {
        self.check_same_dimension(other)?;
        for (_, &v) in other.data.iter().enumerate() {
            if v == FloatScalar::zero() {
                return Err(VecMathError::DivisionByZero);
            }
        }
        let result: Vec<FloatScalar> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a / b)
            .collect();
        Ok(Vector::new(result))
    }

    pub fn add_scalar(&self, scalar: FloatScalar) -> Vector {
        Vector::new(self.data.iter().map(|v| v + scalar).collect())
    }

    pub fn sub_scalar(&self, scalar: FloatScalar) -> Vector {
        Vector::new(self.data.iter().map(|v| v - scalar).collect())
    }

    pub fn mul_scalar(&self, scalar: FloatScalar) -> Vector {
        Vector::new(self.data.iter().map(|v| v * scalar).collect())
    }

    pub fn div_scalar(&self, scalar: FloatScalar) -> Result<Vector> {
        if scalar == FloatScalar::zero() {
            return Err(VecMathError::DivisionByZero);
        }
        Ok(Vector::new(self.data.iter().map(|v| v / scalar).collect()))
    }

    pub fn scalar_add(scalar: FloatScalar, dim: usize) -> Vector {
        Vector::from_scalar(scalar, dim)
    }
}

impl fmt::Debug for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector({:?})", self.data)
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let components: Vec<String> = self.data
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect();
        write!(f, "[{}]", components.join(", "))
    }
}

impl Index<usize> for Vector {
    type Output = FloatScalar;
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i]
    }
}

impl Add for Vector {
    type Output = Result<Vector>;
    fn add(self, other: Vector) -> Self::Output {
        (&self).add(&other)
    }
}

impl Sub for Vector {
    type Output = Result<Vector>;
    fn sub(self, other: Vector) -> Self::Output {
        (&self).sub(&other)
    }
}

impl Mul<FloatScalar> for Vector {
    type Output = Vector;
    fn mul(self, scalar: FloatScalar) -> Self::Output {
        (&self).mul_scalar(scalar)
    }
}

impl Mul<FloatScalar> for &Vector {
    type Output = Vector;
    fn mul(self, scalar: FloatScalar) -> Self::Output {
        self.mul_scalar(scalar)
    }
}

impl Div<FloatScalar> for Vector {
    type Output = Result<Vector>;
    fn div(self, scalar: FloatScalar) -> Self::Output {
        (&self).div_scalar(scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!((&a).add(&b).unwrap(), Vector::new(vec![5.0, 7.0, 9.0]));
        assert_eq!((&a).sub(&b).unwrap(), Vector::new(vec![-3.0, -3.0, -3.0]));
        assert_eq!(a.mul_scalar(2.0), Vector::new(vec![2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_dim_mismatch() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![1.0, 2.0, 3.0]);
        assert!((&a).add(&b).is_err());
    }

    #[test]
    fn test_validate() {
        let a = Vector::new(vec![1.0, FloatScalar::NAN]);
        assert!(a.validate().is_err());
        let b = Vector::new(vec![1.0, FloatScalar::INFINITY]);
        assert!(b.validate().is_err());
    }
}

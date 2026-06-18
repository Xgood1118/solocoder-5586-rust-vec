use serde::{Deserialize, Serialize};
use num_traits::{Zero, One};
use crate::vector::{Vector, FloatScalar};
use crate::error::{VecMathError, Result};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Matrix4 {
    data: [FloatScalar; 16],
}

impl Matrix4 {
    pub fn from_rows(rows: [[FloatScalar; 4]; 4]) -> Self {
        let mut data = [FloatScalar::zero(); 16];
        for r in 0..4 {
            for c in 0..4 {
                data[r * 4 + c] = rows[r][c];
            }
        }
        Matrix4 { data }
    }

    pub fn from_columns(cols: [[FloatScalar; 4]; 4]) -> Self {
        let mut data = [FloatScalar::zero(); 16];
        for c in 0..4 {
            for r in 0..4 {
                data[r * 4 + c] = cols[c][r];
            }
        }
        Matrix4 { data }
    }

    pub fn identity() -> Self {
        let zero = FloatScalar::zero();
        let one = FloatScalar::one();
        Matrix4::from_rows([
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, one, zero],
            [zero, zero, zero, one],
        ])
    }

    pub fn translation(tx: FloatScalar, ty: FloatScalar, tz: FloatScalar) -> Self {
        let zero = FloatScalar::zero();
        let one = FloatScalar::one();
        Matrix4::from_rows([
            [one, zero, zero, tx],
            [zero, one, zero, ty],
            [zero, zero, one, tz],
            [zero, zero, zero, one],
        ])
    }

    pub fn scaling(sx: FloatScalar, sy: FloatScalar, sz: FloatScalar) -> Self {
        let zero = FloatScalar::zero();
        let one = FloatScalar::one();
        Matrix4::from_rows([
            [sx, zero, zero, zero],
            [zero, sy, zero, zero],
            [zero, zero, sz, zero],
            [zero, zero, zero, one],
        ])
    }

    pub fn rotation_x(angle_rad: FloatScalar) -> Self {
        let zero = FloatScalar::zero();
        let one = FloatScalar::one();
        let cos = angle_rad.cos();
        let sin = angle_rad.sin();
        Matrix4::from_rows([
            [one, zero, zero, zero],
            [zero, cos, -sin, zero],
            [zero, sin, cos, zero],
            [zero, zero, zero, one],
        ])
    }

    pub fn rotation_y(angle_rad: FloatScalar) -> Self {
        let zero = FloatScalar::zero();
        let one = FloatScalar::one();
        let cos = angle_rad.cos();
        let sin = angle_rad.sin();
        Matrix4::from_rows([
            [cos, zero, sin, zero],
            [zero, one, zero, zero],
            [-sin, zero, cos, zero],
            [zero, zero, zero, one],
        ])
    }

    pub fn rotation_z(angle_rad: FloatScalar) -> Self {
        let zero = FloatScalar::zero();
        let one = FloatScalar::one();
        let cos = angle_rad.cos();
        let sin = angle_rad.sin();
        Matrix4::from_rows([
            [cos, -sin, zero, zero],
            [sin, cos, zero, zero],
            [zero, zero, one, zero],
            [zero, zero, zero, one],
        ])
    }

    pub fn get(&self, r: usize, c: usize) -> FloatScalar {
        self.data[r * 4 + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: FloatScalar) {
        self.data[r * 4 + c] = v;
    }

    pub fn mul(&self, other: &Matrix4) -> Matrix4 {
        let mut result = [FloatScalar::zero(); 16];
        for r in 0..4 {
            for c in 0..4 {
                let mut sum = FloatScalar::zero();
                for k in 0..4 {
                    sum += self.get(r, k) * other.get(k, c);
                }
                result[r * 4 + c] = sum;
            }
        }
        Matrix4 { data: result }
    }

    pub fn transform_point(&self, point: &Vector) -> Result<Vector> {
        point.check_dimension(3)?;
        let x = point[0];
        let y = point[1];
        let z = point[2];
        let one = FloatScalar::one();
        let nx = self.get(0, 0) * x + self.get(0, 1) * y + self.get(0, 2) * z + self.get(0, 3) * one;
        let ny = self.get(1, 0) * x + self.get(1, 1) * y + self.get(1, 2) * z + self.get(1, 3) * one;
        let nz = self.get(2, 0) * x + self.get(2, 1) * y + self.get(2, 2) * z + self.get(2, 3) * one;
        let w = self.get(3, 0) * x + self.get(3, 1) * y + self.get(3, 2) * z + self.get(3, 3) * one;
        if w == FloatScalar::zero() {
            return Err(VecMathError::DivisionByZero);
        }
        let w_inv = FloatScalar::one() / w;
        Ok(Vector::new(vec![nx * w_inv, ny * w_inv, nz * w_inv]))
    }

    pub fn transform_direction(&self, direction: &Vector) -> Result<Vector> {
        direction.check_dimension(3)?;
        let x = direction[0];
        let y = direction[1];
        let z = direction[2];
        let nx = self.get(0, 0) * x + self.get(0, 1) * y + self.get(0, 2) * z;
        let ny = self.get(1, 0) * x + self.get(1, 1) * y + self.get(1, 2) * z;
        let nz = self.get(2, 0) * x + self.get(2, 1) * y + self.get(2, 2) * z;
        Ok(Vector::new(vec![nx, ny, nz]))
    }

    pub fn to_position_transform(rotation: &Matrix4, translation: (FloatScalar, FloatScalar, FloatScalar)) -> Self {
        let t = Self::translation(translation.0, translation.1, translation.2);
        rotation.mul(&t)
    }

    pub fn to_direction_transform(rotation: &Matrix4) -> Self {
        let mut m = rotation.clone();
        m.set(0, 3, FloatScalar::zero());
        m.set(1, 3, FloatScalar::zero());
        m.set(2, 3, FloatScalar::zero());
        m.set(3, 3, FloatScalar::one());
        m
    }
}

impl std::ops::Mul for &Matrix4 {
    type Output = Matrix4;
    fn mul(self, other: &Matrix4) -> Self::Output {
        self.mul(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = Matrix4::identity();
        assert_eq!(id.get(0, 0), FloatScalar::one());
        assert_eq!(id.get(1, 1), FloatScalar::one());
        assert_eq!(id.get(2, 2), FloatScalar::one());
        assert_eq!(id.get(3, 3), FloatScalar::one());
    }

    #[test]
    fn test_translation() {
        let t = Matrix4::translation(
            num_traits::cast(1.0).unwrap(),
            num_traits::cast(2.0).unwrap(),
            num_traits::cast(3.0).unwrap(),
        );
        let p = Vector::new(vec![0.0, 0.0, 0.0]);
        let r = t.transform_point(&p).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 2.0).abs() < 1e-10);
        assert!((r[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_direction_no_translation() {
        let t = Matrix4::translation(
            num_traits::cast(100.0).unwrap(),
            num_traits::cast(200.0).unwrap(),
            num_traits::cast(300.0).unwrap(),
        );
        let dir_m = Matrix4::to_direction_transform(&t);
        let dir = Vector::new(vec![1.0, 0.0, 0.0]);
        let r = dir_m.transform_direction(&dir).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!(r[1].abs() < 1e-10);
        assert!(r[2].abs() < 1e-10);
    }

    #[test]
    fn test_column_major_translation() {
        let zero = FloatScalar::zero();
        let one = FloatScalar::one();
        let tx: FloatScalar = num_traits::cast(1.0).unwrap();
        let ty: FloatScalar = num_traits::cast(2.0).unwrap();
        let tz: FloatScalar = num_traits::cast(3.0).unwrap();

        let m = Matrix4::from_columns([
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, one, zero],
            [tx, ty, tz, one],
        ]);

        let p = Vector::new(vec![0.0, 0.0, 0.0]);
        let r = m.transform_point(&p).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 2.0).abs() < 1e-10);
        assert!((r[2] - 3.0).abs() < 1e-10);

        let p2 = Vector::new(vec![5.0, 6.0, 7.0]);
        let r2 = m.transform_point(&p2).unwrap();
        assert!((r2[0] - 6.0).abs() < 1e-10);
        assert!((r2[1] - 8.0).abs() < 1e-10);
        assert!((r2[2] - 10.0).abs() < 1e-10);
    }
}

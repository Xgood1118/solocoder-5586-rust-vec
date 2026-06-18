pub mod error;
pub mod vector;
pub mod simd;
pub mod algebra;
pub mod interpolation;
pub mod matrix;
pub mod stats;
pub mod io;
pub mod cli;

pub use error::{VecMathError, Result};
pub use vector::{Vector, FloatScalar};
pub use algebra::{NormType, DistanceType, AngleUnit, PairwiseResult};
pub use interpolation::BezierType;
pub use matrix::Matrix4;
pub use stats::{VectorStatistics, BatchStatistics, HistogramBin};
pub use io::{InputFormat, JsonlRecord};
pub use cli::{Cli, Commands, ArithmeticOp, AlgebraOp, BezierArg, NormArg, DistanceArg, AngleArg, FormatArg, TransformMode, BatchOp};

use rayon::prelude::*;
use num_traits::{cast, Zero};
use algebra::{norm, normalize, dot, distance, cosine_similarity};
use io::{read_vectors, write_vectors, parse_vector_from_cli, InputFormat as IFmt};
use stats::compute_batch_statistics;
use interpolation::{bezier_linear, bezier_quadratic, bezier_cubic, bezier_spherical, sample_bezier};

pub fn run_cli(cli: &cli::Cli) -> Result<()> {
    if let Some(w) = cli.workers {
        if w == 0 {
            return Err(VecMathError::InvalidWorkerCount(w));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(w)
            .build_global()
            .map_err(|e| VecMathError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    }

    match &cli.command {
        Commands::Arithmetic { .. } => handle_arithmetic(&cli.command),
        Commands::Algebra { .. } => handle_algebra(&cli.command),
        Commands::Interpolate { .. } => handle_interpolate(&cli.command),
        Commands::Transform { .. } => handle_transform(&cli.command),
        Commands::Batch { .. } => handle_batch(&cli.command),
        Commands::Stats { .. } => handle_stats(&cli.command),
    }
}

fn is_file_path(s: &str) -> bool {
    std::path::Path::new(s).exists()
}

fn resolve_format(fmt: Option<cli::FormatArg>) -> Option<IFmt> {
    fmt.map(|f| f.into())
}

fn resolve_vectors_or_file(s: &str, fmt: Option<cli::FormatArg>) -> Result<Vec<Vector>> {
    if is_file_path(s) {
        read_vectors(s, resolve_format(fmt))
    } else {
        Ok(vec![parse_vector_from_cli(s)?])
    }
}

fn handle_arithmetic(cmd: &Commands) -> Result<()> {
    if let Commands::Arithmetic { scalar, op, a, b, output, input_format, output_format } = cmd {
        let a_vecs = resolve_vectors_or_file(a, *input_format)?;
        let ifmt = resolve_format(*output_format);

        match op {
            ArithmeticOp::Add => {
                if *scalar {
                    let s: FloatScalar = b.parse().map_err(|_| VecMathError::ParseError(format!("cannot parse scalar '{}'", b)))?;
                    let result: Vec<Vector> = a_vecs.par_iter().map(|v| v.add_scalar(s)).collect();
                    write_output(output, &result, ifmt)?;
                } else {
                    let b_vecs = resolve_vectors_or_file(b, *input_format)?;
                    if a_vecs.len() == 1 && b_vecs.len() == 1 {
                        let r = a_vecs[0].add(&b_vecs[0])?;
                        write_output(output, &[r], ifmt)?;
                    } else if a_vecs.len() == b_vecs.len() {
                        let result: Result<Vec<Vector>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                            .map(|(av, bv)| av.add(bv)).collect();
                        write_output(output, &result?, ifmt)?;
                    } else if b_vecs.len() == 1 {
                        let result: Result<Vec<Vector>> = a_vecs.par_iter()
                            .map(|av| av.add(&b_vecs[0])).collect();
                        write_output(output, &result?, ifmt)?;
                    } else {
                        return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                    }
                }
            }
            ArithmeticOp::Sub => {
                if *scalar {
                    let s: FloatScalar = b.parse().map_err(|_| VecMathError::ParseError(format!("cannot parse scalar '{}'", b)))?;
                    let result: Vec<Vector> = a_vecs.par_iter().map(|v| v.sub_scalar(s)).collect();
                    write_output(output, &result, ifmt)?;
                } else {
                    let b_vecs = resolve_vectors_or_file(b, *input_format)?;
                    if a_vecs.len() == 1 && b_vecs.len() == 1 {
                        let r = a_vecs[0].sub(&b_vecs[0])?;
                        write_output(output, &[r], ifmt)?;
                    } else if a_vecs.len() == b_vecs.len() {
                        let result: Result<Vec<Vector>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                            .map(|(av, bv)| av.sub(bv)).collect();
                        write_output(output, &result?, ifmt)?;
                    } else {
                        return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                    }
                }
            }
            ArithmeticOp::Mul => {
                if *scalar {
                    let s: FloatScalar = b.parse().map_err(|_| VecMathError::ParseError(format!("cannot parse scalar '{}'", b)))?;
                    let result: Vec<Vector> = a_vecs.par_iter().map(|v| v.mul_scalar(s)).collect();
                    write_output(output, &result, ifmt)?;
                } else {
                    let b_vecs = resolve_vectors_or_file(b, *input_format)?;
                    if a_vecs.len() == 1 && b_vecs.len() == 1 {
                        let r = a_vecs[0].mul_elementwise(&b_vecs[0])?;
                        write_output(output, &[r], ifmt)?;
                    } else if a_vecs.len() == b_vecs.len() {
                        let result: Result<Vec<Vector>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                            .map(|(av, bv)| av.mul_elementwise(bv)).collect();
                        write_output(output, &result?, ifmt)?;
                    } else {
                        return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                    }
                }
            }
            ArithmeticOp::Div => {
                if *scalar {
                    let s: FloatScalar = b.parse().map_err(|_| VecMathError::ParseError(format!("cannot parse scalar '{}'", b)))?;
                    let result: Result<Vec<Vector>> = a_vecs.par_iter().map(|v| v.div_scalar(s)).collect();
                    write_output(output, &result?, ifmt)?;
                } else {
                    let b_vecs = resolve_vectors_or_file(b, *input_format)?;
                    if a_vecs.len() == 1 && b_vecs.len() == 1 {
                        let r = a_vecs[0].div_elementwise(&b_vecs[0])?;
                        write_output(output, &[r], ifmt)?;
                    } else if a_vecs.len() == b_vecs.len() {
                        let result: Result<Vec<Vector>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                            .map(|(av, bv)| av.div_elementwise(bv)).collect();
                        write_output(output, &result?, ifmt)?;
                    } else {
                        return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                    }
                }
            }
        }
        Ok(())
    } else { Ok(()) }
}

fn handle_algebra(cmd: &Commands) -> Result<()> {
    if let Commands::Algebra { op, a, b, norm_type, distance_type, angle_unit, output, input_format } = cmd {
        let a_vecs = resolve_vectors_or_file(a, *input_format)?;
        let nt = norm_type.map(|n| n.into()).unwrap_or(algebra::NormType::L2);
        let dt = distance_type.map(|d| d.into()).unwrap_or(algebra::DistanceType::Euclidean);
        let au = angle_unit.map(|u| u.into()).unwrap_or(algebra::AngleUnit::Radians);

        match op {
            AlgebraOp::Dot => {
                let b_vec = parse_second_vec(b, *input_format)?.ok_or_else(|| VecMathError::ParseError("dot product requires second vector".to_string()))?;
                if a_vecs.len() == 1 {
                    let r = dot(&a_vecs[0], &b_vec)?;
                    println!("{}", r);
                } else {
                    let results: Result<Vec<FloatScalar>> = a_vecs.par_iter()
                        .map(|v| dot(v, &b_vec)).collect();
                    if let Some(o) = output {
                        write_scalars_json(o, &results?)?;
                    } else {
                        for r in results? { println!("{}", r); }
                    }
                }
            }
            AlgebraOp::Cross => {
                let b_vec = parse_second_vec(b, *input_format)?.ok_or_else(|| VecMathError::ParseError("cross product requires second vector".to_string()))?;
                let result: Result<Vec<Vector>> = a_vecs.par_iter()
                    .map(|v| algebra::cross(v, &b_vec)).collect();
                let result = result?;
                write_output(output, &result, resolve_format(None))?;
            }
            AlgebraOp::Norm => {
                if a_vecs.len() == 1 {
                    let r = norm(&a_vecs[0], nt);
                    println!("{}", r);
                } else {
                    let results: Vec<FloatScalar> = a_vecs.par_iter().map(|v| norm(v, nt)).collect();
                    if let Some(o) = output {
                        write_scalars_json(o, &results)?;
                    } else {
                        for r in results { println!("{}", r); }
                    }
                }
            }
            AlgebraOp::Normalize => {
                let result: Result<Vec<Vector>> = a_vecs.par_iter().map(|v| normalize(v)).collect();
                let result = result?;
                write_output(output, &result, resolve_format(None))?;
            }
            AlgebraOp::Distance => {
                let b_vec = parse_second_vec(b, *input_format)?.ok_or_else(|| VecMathError::ParseError("distance requires second vector".to_string()))?;
                if a_vecs.len() == 1 {
                    let r = distance(&a_vecs[0], &b_vec, dt)?;
                    println!("{}", r);
                } else {
                    let results: Result<Vec<FloatScalar>> = a_vecs.par_iter()
                        .map(|v| distance(v, &b_vec, dt)).collect();
                    if let Some(o) = output {
                        write_scalars_json(o, &results?)?;
                    } else {
                        for r in results? { println!("{}", r); }
                    }
                }
            }
            AlgebraOp::Similarity => {
                let b_vec = parse_second_vec(b, *input_format)?.ok_or_else(|| VecMathError::ParseError("cosine similarity requires second vector".to_string()))?;
                if a_vecs.len() == 1 {
                    let r = cosine_similarity(&a_vecs[0], &b_vec)?;
                    println!("{}", r);
                } else {
                    let results: Result<Vec<FloatScalar>> = a_vecs.par_iter()
                        .map(|v| cosine_similarity(v, &b_vec)).collect();
                    if let Some(o) = output {
                        write_scalars_json(o, &results?)?;
                    } else {
                        for r in results? { println!("{}", r); }
                    }
                }
            }
            AlgebraOp::Angle => {
                let b_vec = parse_second_vec(b, *input_format)?.ok_or_else(|| VecMathError::ParseError("angle requires second vector".to_string()))?;
                if a_vecs.len() == 1 {
                    let r = algebra::angle(&a_vecs[0], &b_vec, au)?;
                    println!("{}", r);
                } else {
                    let results: Result<Vec<FloatScalar>> = a_vecs.par_iter()
                        .map(|v| algebra::angle(v, &b_vec, au)).collect();
                    if let Some(o) = output {
                        write_scalars_json(o, &results?)?;
                    } else {
                        for r in results? { println!("{}", r); }
                    }
                }
            }
            AlgebraOp::Project => {
                let b_vec = parse_second_vec(b, *input_format)?.ok_or_else(|| VecMathError::ParseError("projection requires second vector".to_string()))?;
                let result: Result<Vec<Vector>> = a_vecs.par_iter()
                    .map(|v| algebra::project(v, &b_vec)).collect();
                let result = result?;
                write_output(output, &result, resolve_format(None))?;
            }
            AlgebraOp::Reject => {
                let b_vec = parse_second_vec(b, *input_format)?.ok_or_else(|| VecMathError::ParseError("rejection requires second vector".to_string()))?;
                let result: Result<Vec<Vector>> = a_vecs.par_iter()
                    .map(|v| algebra::reject(v, &b_vec)).collect();
                let result = result?;
                write_output(output, &result, resolve_format(None))?;
            }
            AlgebraOp::PairwiseDot => {
                let results = algebra::pairwise_dot(&a_vecs)?;
                let json = serde_json::to_string_pretty(&results)?;
                if let Some(o) = output { std::fs::write(o, json)?; } else { println!("{}", json); }
            }
            AlgebraOp::PairwiseDistance => {
                let results = algebra::pairwise_distance(&a_vecs, dt)?;
                let json = serde_json::to_string_pretty(&results)?;
                if let Some(o) = output { std::fs::write(o, json)?; } else { println!("{}", json); }
            }
        }
        Ok(())
    } else { Ok(()) }
}

fn parse_second_vec(b: &Option<String>, fmt: Option<cli::FormatArg>) -> Result<Option<Vector>> {
    match b {
        Some(s) => {
            let v = if is_file_path(s) {
                let mut vs = read_vectors(s, resolve_format(fmt))?;
                if vs.is_empty() { return Err(VecMathError::EmptyVector); }
                vs.remove(0)
            } else {
                parse_vector_from_cli(s)?
            };
            Ok(Some(v))
        }
        None => Ok(None),
    }
}

fn write_output(output: &Option<String>, vectors: &[Vector], fmt: Option<IFmt>) -> Result<()> {
    match output {
        Some(p) => write_vectors(p, vectors, fmt),
        None => {
            for v in vectors {
                println!("{}", v);
            }
            Ok(())
        }
    }
}

fn write_scalars_json(path: &str, values: &[FloatScalar]) -> Result<()> {
    let json = serde_json::to_string_pretty(&values)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn handle_interpolate(cmd: &Commands) -> Result<()> {
    if let Commands::Interpolate { bezier, samples, points, t, output, output_format } = cmd {
        let bt: BezierType = (*bezier).into();
        let control_points: Result<Vec<Vector>> = points.iter().map(|p| parse_vector_from_cli(p)).collect();
        let control_points = control_points?;

        let result_vectors = if let Some(t_str) = t {
            let t_val: f64 = t_str.parse().map_err(|_| VecMathError::InvalidInterpolationParameter { t: t_str.clone() })?;
            let t_fs: FloatScalar = cast(t_val).unwrap();
            let v = match bt {
                BezierType::Linear => bezier_linear(&control_points[0], &control_points[1], t_fs)?,
                BezierType::Quadratic => bezier_quadratic(&control_points[0], &control_points[1], &control_points[2], t_fs)?,
                BezierType::Cubic => bezier_cubic(&control_points[0], &control_points[1], &control_points[2], &control_points[3], t_fs)?,
                BezierType::Spherical => bezier_spherical(&control_points[0], &control_points[1], t_fs)?,
            };
            vec![v]
        } else {
            sample_bezier(&control_points, bt, *samples)?
        };

        write_output(output, &result_vectors, resolve_format(*output_format))?;
        Ok(())
    } else { Ok(()) }
}

fn parse_matrix(matrix_str: &str) -> Result<Matrix4> {
    if is_file_path(matrix_str) {
        let content = std::fs::read_to_string(matrix_str)?;
        let m: Vec<Vec<FloatScalar>> = serde_json::from_str(&content)?;
        if m.len() != 4 || m.iter().any(|r| r.len() != 4) {
            return Err(VecMathError::MatrixSizeMismatch { expected: 4, rows: m.len(), cols: m.first().map(|r| r.len()).unwrap_or(0) });
        }
        let mut rows = [[FloatScalar::zero(); 4]; 4];
        for r in 0..4 { for c in 0..4 { rows[r][c] = m[r][c]; } }
        Ok(Matrix4::from_rows(rows))
    } else {
        let vals: Vec<FloatScalar> = matrix_str.split(',').map(|s| s.trim().parse()).collect::<std::result::Result<_, _>>()
            .map_err(|e| VecMathError::ParseError(format!("matrix parse error: {}", e)))?;
        if vals.len() != 16 {
            return Err(VecMathError::MatrixSizeMismatch { expected: 4, rows: 0, cols: vals.len() });
        }
        let mut rows = [[FloatScalar::zero(); 4]; 4];
        for r in 0..4 { for c in 0..4 { rows[r][c] = vals[r * 4 + c]; } }
        Ok(Matrix4::from_rows(rows))
    }
}

fn handle_transform(cmd: &Commands) -> Result<()> {
    if let Commands::Transform { mode, matrix, input, output, input_format, output_format } = cmd {
        let m = parse_matrix(matrix)?;
        let a_vecs = resolve_vectors_or_file(input, *input_format)?;
        let ifmt = resolve_format(*output_format);

        let result: Result<Vec<Vector>> = match mode {
            TransformMode::Position => a_vecs.par_iter().map(|v| m.transform_point(v)).collect(),
            TransformMode::Direction => a_vecs.par_iter().map(|v| m.transform_direction(v)).collect(),
        };
        let result = result?;
        write_output(output, &result, ifmt)?;
        Ok(())
    } else { Ok(()) }
}

fn handle_batch(cmd: &Commands) -> Result<()> {
    if let Commands::Batch { op, input, input2, output, input_format, output_format, norm_type, distance_type, scalar, points: _points } = cmd {
        let a_vecs = read_vectors(input, resolve_format(*input_format))?;
        let ifmt = resolve_format(*output_format);
        let nt = norm_type.map(|n| n.into()).unwrap_or(algebra::NormType::L2);
        let dt = distance_type.map(|d| d.into()).unwrap_or(algebra::DistanceType::Euclidean);

        match op {
            BatchOp::Normalize => {
                let result: Result<Vec<Vector>> = a_vecs.par_iter().map(|v| normalize(v)).collect();
                write_vectors(output, &result?, ifmt)?;
            }
            BatchOp::Norm => {
                let results: Vec<FloatScalar> = a_vecs.par_iter().map(|v| norm(v, nt)).collect();
                write_scalars_json(output, &results)?;
            }
            BatchOp::Add => {
                let b_vecs = read_vectors(input2.as_ref().ok_or_else(|| VecMathError::ParseError("batch add requires --input2".to_string()))?, resolve_format(*input_format))?;
                if a_vecs.len() != b_vecs.len() {
                    return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                }
                let result: Result<Vec<Vector>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                    .map(|(a, b)| a.add(b)).collect();
                write_vectors(output, &result?, ifmt)?;
            }
            BatchOp::Sub => {
                let b_vecs = read_vectors(input2.as_ref().ok_or_else(|| VecMathError::ParseError("batch sub requires --input2".to_string()))?, resolve_format(*input_format))?;
                if a_vecs.len() != b_vecs.len() {
                    return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                }
                let result: Result<Vec<Vector>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                    .map(|(a, b)| a.sub(b)).collect();
                write_vectors(output, &result?, ifmt)?;
            }
            BatchOp::MulScalar => {
                let s: FloatScalar = scalar.as_ref().ok_or_else(|| VecMathError::ParseError("--scalar required".to_string()))?
                    .parse().map_err(|_| VecMathError::ParseError("scalar parse error".to_string()))?;
                let result: Vec<Vector> = a_vecs.par_iter().map(|v| v.mul_scalar(s)).collect();
                write_vectors(output, &result, ifmt)?;
            }
            BatchOp::DivScalar => {
                let s: FloatScalar = scalar.as_ref().ok_or_else(|| VecMathError::ParseError("--scalar required".to_string()))?
                    .parse().map_err(|_| VecMathError::ParseError("scalar parse error".to_string()))?;
                let result: Result<Vec<Vector>> = a_vecs.par_iter().map(|v| v.div_scalar(s)).collect();
                write_vectors(output, &result?, ifmt)?;
            }
            BatchOp::Distance => {
                let b_vecs = read_vectors(input2.as_ref().ok_or_else(|| VecMathError::ParseError("batch distance requires --input2".to_string()))?, resolve_format(*input_format))?;
                if a_vecs.len() != b_vecs.len() {
                    return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                }
                let results: Result<Vec<FloatScalar>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                    .map(|(a, b)| distance(a, b, dt)).collect();
                write_scalars_json(output, &results?)?;
            }
            BatchOp::Dot => {
                let b_vecs = read_vectors(input2.as_ref().ok_or_else(|| VecMathError::ParseError("batch dot requires --input2".to_string()))?, resolve_format(*input_format))?;
                if a_vecs.len() != b_vecs.len() {
                    return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                }
                let results: Result<Vec<FloatScalar>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                    .map(|(a, b)| dot(a, b)).collect();
                write_scalars_json(output, &results?)?;
            }
            BatchOp::Similarity => {
                let b_vecs = read_vectors(input2.as_ref().ok_or_else(|| VecMathError::ParseError("batch similarity requires --input2".to_string()))?, resolve_format(*input_format))?;
                if a_vecs.len() != b_vecs.len() {
                    return Err(VecMathError::DimensionMismatch { expected: a_vecs.len(), got: b_vecs.len() });
                }
                let results: Result<Vec<FloatScalar>> = a_vecs.par_iter().zip(b_vecs.par_iter())
                    .map(|(a, b)| cosine_similarity(a, b)).collect();
                write_scalars_json(output, &results?)?;
            }
            BatchOp::Histogram => {
                println!("Use 'stats' subcommand for histogram per dimension.");
            }
        }
        Ok(())
    } else { Ok(()) }
}

fn handle_stats(cmd: &Commands) -> Result<()> {
    if let Commands::Stats { input, quantiles, bins, input_format, output } = cmd {
        let vectors = read_vectors(input, resolve_format(*input_format))?;
        let batch = compute_batch_statistics(&vectors, quantiles, *bins)?;
        let json = serde_json::to_string_pretty(&batch)?;
        match output {
            Some(o) => std::fs::write(o, json)?,
            None => println!("{}", json),
        }
        Ok(())
    } else { Ok(()) }
}

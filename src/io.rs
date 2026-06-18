use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::fs::File;
use crate::vector::{Vector, FloatScalar};
use crate::error::{VecMathError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputFormat {
    Jsonl,
    Csv,
    Binary,
}

impl InputFormat {
    pub fn from_filename(path: &str) -> Result<Self> {
        let lower = path.to_lowercase();
        if lower.ends_with(".jsonl") || lower.ends_with(".json") {
            Ok(InputFormat::Jsonl)
        } else if lower.ends_with(".csv") {
            Ok(InputFormat::Csv)
        } else if lower.ends_with(".bin") || lower.ends_with(".binary") {
            Ok(InputFormat::Binary)
        } else {
            Err(VecMathError::UnsupportedFormat(lower))
        }
    }

    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "jsonl" | "json" => Ok(InputFormat::Jsonl),
            "csv" => Ok(InputFormat::Csv),
            "binary" | "bin" => Ok(InputFormat::Binary),
            _ => Err(VecMathError::UnsupportedFormat(s.to_string())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonlRecord {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<usize>,
    pub vector: Vec<FloatScalar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

pub fn read_vectors(path: &str, format: Option<InputFormat>) -> Result<Vec<Vector>> {
    let fmt = match format {
        Some(f) => f,
        None => InputFormat::from_filename(path)?,
    };

    match fmt {
        InputFormat::Jsonl => read_jsonl(path),
        InputFormat::Csv => read_csv(path),
        InputFormat::Binary => read_binary(path),
    }
}

pub fn write_vectors(path: &str, vectors: &[Vector], format: Option<InputFormat>) -> Result<()> {
    let fmt = match format {
        Some(f) => f,
        None => InputFormat::from_filename(path)?,
    };

    match fmt {
        InputFormat::Jsonl => write_jsonl(path, vectors),
        InputFormat::Csv => write_csv(path, vectors),
        InputFormat::Binary => write_binary(path, vectors),
    }
}

fn read_jsonl(path: &str) -> Result<Vec<Vector>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    if contents.starts_with('\u{FEFF}') {
        contents.drain(..3);
    }

    let mut vectors = Vec::new();
    for (line_idx, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Ok(record) = serde_json::from_str::<JsonlRecord>(line) {
            let v = Vector::new(record.vector);
            if let Err(e) = v.validate() {
                eprintln!("Warning line {}: {}", line_idx + 1, e);
                continue;
            }
            vectors.push(v);
        } else if let Ok(arr) = serde_json::from_str::<Vec<FloatScalar>>(line) {
            let v = Vector::new(arr);
            if let Err(e) = v.validate() {
                eprintln!("Warning line {}: {}", line_idx + 1, e);
                continue;
            }
            vectors.push(v);
        } else {
            return Err(VecMathError::JsonError(
                serde_json::from_str::<serde_json::Value>(line).unwrap_err()
            ));
        }
    }
    Ok(vectors)
}

fn write_jsonl(path: &str, vectors: &[Vector]) -> Result<()> {
    let mut file = File::create(path)?;
    for (i, v) in vectors.iter().enumerate() {
        let record = JsonlRecord {
            id: Some(i),
            vector: v.data().to_vec(),
            metadata: None,
        };
        let line = serde_json::to_string(&record)?;
        writeln!(file, "{}", line)?;
    }
    Ok(())
}

fn read_csv(path: &str) -> Result<Vec<Vector>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;

    let mut vectors = Vec::new();
    for (row_idx, result) in rdr.records().enumerate() {
        let record = result?;
        let mut data: Vec<FloatScalar> = Vec::with_capacity(record.len());
        let mut valid = true;
        for field in record.iter() {
            let field = field.trim();
            match field.parse::<FloatScalar>() {
                Ok(v) => data.push(v),
                Err(_) => {
                    eprintln!("Warning line {}: cannot parse '{}'", row_idx + 1, field);
                    valid = false;
                    break;
                }
            }
        }
        if !valid { continue; }
        let v = Vector::new(data);
        if let Err(e) = v.validate() {
            eprintln!("Warning line {}: {}", row_idx + 1, e);
            continue;
        }
        vectors.push(v);
    }
    Ok(vectors)
}

fn write_csv(path: &str, vectors: &[Vector]) -> Result<()> {
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(path)?;

    for v in vectors {
        let row: Vec<String> = v.data().iter().map(|x| format!("{}", x)).collect();
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

const BINARY_MAGIC: u32 = 0x5645434D;
const BINARY_VERSION: u32 = 1;

fn read_binary(path: &str) -> Result<Vec<Vector>> {
    use std::io::Read;
    use bytemuck::cast_slice;

    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    if buf.len() < 12 {
        return Err(VecMathError::ParseError("binary file too short".to_string()));
    }

    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let count = u32::from_le_bytes(buf[8..12].try_into().unwrap());

    if magic != BINARY_MAGIC {
        return Err(VecMathError::ParseError(format!(
            "invalid binary magic: expected 0x{:X}, got 0x{:X}", BINARY_MAGIC, magic
        )));
    }
    if version != BINARY_VERSION {
        return Err(VecMathError::ParseError(format!(
            "unsupported binary version: expected {}, got {}", BINARY_VERSION, version
        )));
    }

    let mut offset = 12usize;
    let mut vectors = Vec::with_capacity(count as usize);

    for _ in 0..count {
        if offset + 4 > buf.len() {
            return Err(VecMathError::ParseError("unexpected EOF at dim".to_string()));
        }
        let dim = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let bytes_len = dim * std::mem::size_of::<FloatScalar>();
        if offset + bytes_len > buf.len() {
            return Err(VecMathError::ParseError("unexpected EOF at vector data".to_string()));
        }

        let bytes = &buf[offset..offset + bytes_len];
        let data: Vec<FloatScalar> = if cfg!(target_endian = "little") {
            cast_slice::<u8, FloatScalar>(bytes).to_vec()
        } else {
            bytes.chunks_exact(std::mem::size_of::<FloatScalar>())
                .map(|chunk| {
                    let mut arr = [0u8; 8];
                    for (i, &b) in chunk.iter().enumerate() {
                        arr[i] = b;
                    }
                    FloatScalar::from_le_bytes(arr)
                })
                .collect()
        };
        offset += bytes_len;

        let v = Vector::new(data);
        if let Err(e) = v.validate() {
            eprintln!("Warning: {}", e);
            continue;
        }
        vectors.push(v);
    }

    Ok(vectors)
}

fn write_binary(path: &str, vectors: &[Vector]) -> Result<()> {
    use std::io::Write;
    use bytemuck::cast_slice;

    let mut file = File::create(path)?;

    file.write_all(&BINARY_MAGIC.to_le_bytes())?;
    file.write_all(&BINARY_VERSION.to_le_bytes())?;
    file.write_all(&(vectors.len() as u32).to_le_bytes())?;

    for v in vectors {
        let dim = v.dim() as u32;
        file.write_all(&dim.to_le_bytes())?;
        let bytes = if cfg!(target_endian = "little") {
            cast_slice::<FloatScalar, u8>(v.data()).to_vec()
        } else {
            v.data().iter().flat_map(|&x| x.to_le_bytes()).collect()
        };
        file.write_all(&bytes)?;
    }

    file.flush()?;
    Ok(())
}

pub fn parse_vector_from_cli(arg: &str) -> Result<Vector> {
    let trimmed = arg.trim();
    if let Ok(v) = parse_bracketed_vector(trimmed) {
        return Ok(v);
    }
    if let Ok(v) = parse_space_separated(trimmed) {
        return Ok(v);
    }
    if let Ok(v) = parse_comma_separated(trimmed) {
        return Ok(v);
    }
    Err(VecMathError::ParseError(format!(
        "cannot parse vector from '{}'. Use [1,2,3] or '1 2 3' or '1,2,3'",
        arg
    )))
}

fn parse_bracketed_vector(s: &str) -> Result<Vector> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(VecMathError::ParseError("not bracketed".to_string()));
    }
    let inner = &s[1..s.len() - 1];
    parse_comma_separated(inner)
}

fn parse_space_separated(s: &str) -> Result<Vector> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return Err(VecMathError::ParseError("empty string".to_string()));
    }
    let mut data = Vec::with_capacity(parts.len());
    for p in parts {
        match p.parse::<FloatScalar>() {
            Ok(v) => data.push(v),
            Err(_) => return Err(VecMathError::ParseError(format!("cannot parse '{}'", p))),
        }
    }
    Ok(Vector::new(data))
}

fn parse_comma_separated(s: &str) -> Result<Vector> {
    let parts: Vec<&str> = s.split(',').map(|x| x.trim()).collect();
    if parts.is_empty() {
        return Err(VecMathError::ParseError("empty string".to_string()));
    }
    let mut data = Vec::with_capacity(parts.len());
    for p in parts {
        if p.is_empty() { continue; }
        match p.parse::<FloatScalar>() {
            Ok(v) => data.push(v),
            Err(_) => return Err(VecMathError::ParseError(format!("cannot parse '{}'", p))),
        }
    }
    if data.is_empty() {
        return Err(VecMathError::ParseError("empty vector".to_string()));
    }
    Ok(Vector::new(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vector_bracketed() {
        let v = parse_vector_from_cli("[1, 2, 3]").unwrap();
        assert_eq!(v.dim(), 3);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_parse_vector_space() {
        let v = parse_vector_from_cli("1.0 2.0 3.0").unwrap();
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn test_format_detection() {
        assert!(matches!(
            InputFormat::from_filename("vectors.jsonl").unwrap(),
            InputFormat::Jsonl
        ));
        assert!(matches!(
            InputFormat::from_filename("vectors.csv").unwrap(),
            InputFormat::Csv
        ));
        assert!(matches!(
            InputFormat::from_filename("vectors.bin").unwrap(),
            InputFormat::Binary
        ));
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VecMathError {
    #[error("E001: Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("E002: Zero vector encountered: cannot normalize a zero-length vector")]
    ZeroVector,

    #[error("E003: Invalid dimension for cross product: only 3D vectors supported, got {dim}")]
    InvalidCrossDimension { dim: usize },

    #[error("E004: Division by zero: scalar divisor is zero")]
    DivisionByZero,

    #[error("E005: Input contains NaN value at index {index}")]
    NaNInInput { index: usize },

    #[error("E006: Input contains Inf value at index {index}")]
    InfInInput { index: usize },

    #[error("E007: Invalid parameter: {param} = {value}, {reason}")]
    InvalidParameter {
        param: String,
        value: String,
        reason: String,
    },

    #[error("E008: Empty vector: operation requires non-empty vector")]
    EmptyVector,

    #[error("E009: Not enough control points for Bezier curve: expected {expected}, got {got}")]
    InsufficientControlPoints { expected: usize, got: usize },

    #[error("E010: Invalid interpolation parameter t: must be in [0, 1], got {t}")]
    InvalidInterpolationParameter { t: String },

    #[error("E011: IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("E012: JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("E013: CSV parse error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("E014: Parse error: {0}")]
    ParseError(String),

    #[error("E015: Matrix dimension mismatch for multiplication: ({lrows}, {lcols}) vs ({rrows}, {rcols})")]
    MatrixMulDimensionMismatch {
        lrows: usize,
        lcols: usize,
        rrows: usize,
        rcols: usize,
    },

    #[error("E016: Singular matrix: cannot invert")]
    SingularMatrix,

    #[error("E017: Not enough data points for statistics: expected at least {expected}, got {got}")]
    InsufficientData { expected: usize, got: usize },

    #[error("E018: Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("E019: Matrix size mismatch: expected {expected}x{expected}, got {rows}x{cols}")]
    MatrixSizeMismatch {
        expected: usize,
        rows: usize,
        cols: usize,
    },

    #[error("E020: Worker count must be positive, got {0}")]
    InvalidWorkerCount(usize),

    #[error("E021: Insufficient vectors for pairwise operation: expected at least {expected}, got {got}")]
    InsufficientVectors { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, VecMathError>;

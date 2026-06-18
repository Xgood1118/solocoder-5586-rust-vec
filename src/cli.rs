use clap::{Parser, Subcommand, ValueEnum};
use crate::algebra::{NormType, DistanceType, AngleUnit};
use crate::interpolation::BezierType;
use crate::io::InputFormat;

#[derive(Parser, Debug)]
#[command(name = "vecmath-cli")]
#[command(about = "High-performance vector math CLI tool", long_about = None)]
#[command(version)]
pub struct Cli {
    #[arg(short, long, help = "Number of worker threads for parallel processing")]
    pub workers: Option<usize>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Vector arithmetic operations (add/sub/mul/div)")]
    Arithmetic {
        #[arg(short, long, help = "Enable scalar mode: second operand is a scalar value")]
        scalar: bool,

        #[arg(short = 'p', long, value_enum, help = "Operation type")]
        op: ArithmeticOp,

        #[arg(help = "First vector (e.g. '[1,2,3]') or input file path")]
        a: String,

        #[arg(help = "Second vector, scalar value, or input file path")]
        b: String,

        #[arg(short, long, help = "Output file path (stdout if omitted)")]
        output: Option<String>,

        #[arg(long, value_enum, help = "Input format override")]
        input_format: Option<FormatArg>,

        #[arg(long, value_enum, help = "Output format override")]
        output_format: Option<FormatArg>,
    },

    #[command(about = "Vector algebra operations (dot/cross/norm/normalize/distance/similarity/angle/project)")]
    Algebra {
        #[arg(short = 'p', long, value_enum, help = "Algebra operation type")]
        op: AlgebraOp,

        #[arg(help = "First vector (e.g. '[1,2,3]') or input file path")]
        a: String,

        #[arg(help = "Second vector or input file path (omitted for single-vector ops like norm)")]
        b: Option<String>,

        #[arg(short = 't', long, value_enum, help = "Norm type: l1/l2/linf")]
        norm_type: Option<NormArg>,

        #[arg(short = 'd', long, value_enum, help = "Distance type: euclidean/manhattan/chebyshev")]
        distance_type: Option<DistanceArg>,

        #[arg(short = 'u', long, value_enum, help = "Angle unit: radians/degrees")]
        angle_unit: Option<AngleArg>,

        #[arg(short, long, help = "Output file path (stdout if omitted)")]
        output: Option<String>,

        #[arg(long, value_enum, help = "Input format override")]
        input_format: Option<FormatArg>,
    },

    #[command(about = "Bezier interpolation curves (linear/quadratic/cubic/spherical)")]
    Interpolate {
        #[arg(short, long, value_enum, help = "Bezier type")]
        bezier: BezierArg,

        #[arg(short, long, default_value_t = 10, help = "Number of sample points to output")]
        samples: usize,

        #[arg(long, num_args = 1..=4, help = "Control points as vectors, e.g. '[0,0]' '[1,1]'")]
        points: Vec<String>,

        #[arg(long, help = "Single t value in [0,1] for a single point instead of samples")]
        t: Option<String>,

        #[arg(short, long, help = "Output file path (stdout if omitted)")]
        output: Option<String>,

        #[arg(long, value_enum, help = "Output format override")]
        output_format: Option<FormatArg>,
    },

    #[command(about = "4x4 matrix transforms (position/direction)")]
    Transform {
        #[arg(short, long, value_enum, help = "Transform mode: position or direction")]
        mode: TransformMode,

        #[arg(long, help = "Matrix file (JSON 4x4) or 16 comma-separated values")]
        matrix: String,

        #[arg(help = "Input vector(s) file or a single vector")]
        input: String,

        #[arg(short, long, help = "Output file path")]
        output: Option<String>,

        #[arg(long, value_enum, help = "Input format override")]
        input_format: Option<FormatArg>,

        #[arg(long, value_enum, help = "Output format override")]
        output_format: Option<FormatArg>,
    },

    #[command(about = "Batch parallel processing from file")]
    Batch {
        #[arg(short = 'p', long, value_enum, help = "Operation type for batch processing")]
        op: BatchOp,

        #[arg(short, long, help = "Input file path (jsonl/csv/bin)")]
        input: String,

        #[arg(short = 'x', long, help = "Second input file path for pairwise ops (e.g. distance)")]
        input2: Option<String>,

        #[arg(short, long, help = "Output file path")]
        output: String,

        #[arg(long, value_enum, help = "Input format override")]
        input_format: Option<FormatArg>,

        #[arg(long, value_enum, help = "Output format override")]
        output_format: Option<FormatArg>,

        #[arg(long, value_enum, help = "Norm type: l1/l2/linf")]
        norm_type: Option<NormArg>,

        #[arg(long, value_enum, help = "Distance type: euclidean/manhattan/chebyshev")]
        distance_type: Option<DistanceArg>,

        #[arg(long, help = "Scalar value for scalar operations")]
        scalar: Option<String>,

        #[arg(long, num_args = 1..=4, help = "Control points for batch interpolation")]
        points: Option<Vec<String>>,
    },

    #[command(about = "Compute statistics (mean/variance/std-dev/quantiles/histogram)")]
    Stats {
        #[arg(short, long, help = "Input file path (jsonl/csv/bin)")]
        input: String,

        #[arg(short = 'q', long, num_args = 1.., default_values = ["0.25", "0.5", "0.75", "0.95", "0.99"],
               help = "Quantiles to compute (0-1), e.g. -q 0.25 0.5 0.75")]
        quantiles: Vec<f64>,

        #[arg(short = 'b', long, default_value_t = 10, help = "Number of histogram bins")]
        bins: usize,

        #[arg(long, value_enum, help = "Input format override")]
        input_format: Option<FormatArg>,

        #[arg(short, long, help = "Output JSON file path (stdout JSON if omitted)")]
        output: Option<String>,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum AlgebraOp {
    Dot,
    Cross,
    Norm,
    Normalize,
    Distance,
    Similarity,
    Angle,
    Project,
    Reject,
    PairwiseDot,
    PairwiseDistance,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum BezierArg {
    Linear,
    Quadratic,
    Cubic,
    Spherical,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum NormArg {
    L1,
    L2,
    Linf,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DistanceArg {
    Euclidean,
    Manhattan,
    Chebyshev,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum AngleArg {
    Radians,
    Degrees,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum FormatArg {
    Jsonl,
    Csv,
    Binary,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum TransformMode {
    Position,
    Direction,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum BatchOp {
    Normalize,
    Norm,
    Add,
    Sub,
    MulScalar,
    DivScalar,
    Distance,
    Dot,
    Similarity,
    Histogram,
}

impl From<NormArg> for NormType {
    fn from(n: NormArg) -> Self {
        match n {
            NormArg::L1 => NormType::L1,
            NormArg::L2 => NormType::L2,
            NormArg::Linf => NormType::LInf,
        }
    }
}

impl From<DistanceArg> for DistanceType {
    fn from(d: DistanceArg) -> Self {
        match d {
            DistanceArg::Euclidean => DistanceType::Euclidean,
            DistanceArg::Manhattan => DistanceType::Manhattan,
            DistanceArg::Chebyshev => DistanceType::Chebyshev,
        }
    }
}

impl From<AngleArg> for AngleUnit {
    fn from(a: AngleArg) -> Self {
        match a {
            AngleArg::Radians => AngleUnit::Radians,
            AngleArg::Degrees => AngleUnit::Degrees,
        }
    }
}

impl From<BezierArg> for BezierType {
    fn from(b: BezierArg) -> Self {
        match b {
            BezierArg::Linear => BezierType::Linear,
            BezierArg::Quadratic => BezierType::Quadratic,
            BezierArg::Cubic => BezierType::Cubic,
            BezierArg::Spherical => BezierType::Spherical,
        }
    }
}

impl From<FormatArg> for InputFormat {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Jsonl => InputFormat::Jsonl,
            FormatArg::Csv => InputFormat::Csv,
            FormatArg::Binary => InputFormat::Binary,
        }
    }
}

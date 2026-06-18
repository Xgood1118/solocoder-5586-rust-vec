use clap::Parser;
use vecmath_cli::{Cli, run_cli};

fn main() {
    let cli = Cli::parse();
    match run_cli(&cli) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

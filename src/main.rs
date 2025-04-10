#![feature(portable_simd)]
#![expect(clippy::cast_precision_loss, clippy::missing_panics_doc)]

// NOTE: Verification:
// ||Ax - b||         < e при dim < 10.000
// ||Ax - b|| / ||b|| < e при dim >= 10.000

use clap::Parser;
use color_eyre::eyre::Report;
use nalgebra_sparse::{io, CsrMatrix};
use rand::Rng;

pub mod benchmark;
pub mod cli;

pub const FLOAT_TOLERANCE: f64 = 10e-4;
pub const MAX_BICGSTAB_ITERATIONS: usize = 1_000_000;
pub const MAX_RAYON_THREADS: usize = 100;

fn main() -> Result<(), Report> {
    color_eyre::install()?;
    install_tracing()?;
    let mut rng = rand::rng();
    let options = cli::Options::parse();
    let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(options.matrix_path)?;

    let csr_matrix = CsrMatrix::from(&coo_matrix);
    let vector = (0..csr_matrix.nrows())
        .map(|_| rng.random::<f64>())
        .collect::<Vec<_>>();
    dbg!(&vector);

    // let _ = benchmark::run_benchmark(
    //     &csr_matrix,
    //     &vector,
    //     FLOAT_TOLERANCE,
    //     MAX_BICGSTAB_ITERATIONS,
    //     MAX_RAYON_THREADS,
    // );

    Ok(())
}

fn install_tracing() -> Result<(), Report> {
    use tracing_error::ErrorLayer;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let filter_layer = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?;
    let format_layer = fmt::layer()
        .pretty()
        .without_time()
        .with_writer(std::io::stderr);

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(format_layer)
        .with(ErrorLayer::default())
        .try_init()?;

    Ok(())
}

use bmstu_os2_matrix_solver::{benchmark, FLOAT_TOLERANCE, MAX_ITERATIONS};
use clap::Parser;
use color_eyre::eyre::Report;
use nalgebra_sparse::{io, CsrMatrix};

pub mod cli;

fn main() -> Result<(), Report> {
    color_eyre::install()?;
    install_tracing()?;

    let options = cli::Options::parse();
    let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(options.matrix_path)?;
    let csr_matrix = CsrMatrix::from(&coo_matrix);
    let rhs = bmstu_os2_matrix_solver::io::load_vector_from_matrix_market_file(options.rhs_path)?;

    let results = benchmark::run_benchmark(
        &csr_matrix,
        &rhs,
        FLOAT_TOLERANCE,
        MAX_ITERATIONS,
        options.min_threads,
        options.max_threads,
    )?;

    results.create_plots().show();

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

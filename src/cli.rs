use clap::builder::styling::AnsiColor::{Blue, BrightBlue, Magenta};
use clap::builder::Styles;
use clap::Parser;
use std::path::PathBuf;

const STYLES: Styles = Styles::styled()
    .usage(Magenta.on_default().bold())
    .literal(BrightBlue.on_default().bold())
    .placeholder(Blue.on_default().bold())
    .header(Magenta.on_default().bold());

#[derive(Parser, Clone, Debug)]
#[command(styles(STYLES))]
pub struct Options {
    /// Path to an .mtx file containing the target matrix.
    pub matrix_path: PathBuf,

    /// Path to an .mtx file containing the target RHS.
    pub rhs_path: PathBuf,

    /// From what amount of threads to start benchmarking.
    #[arg(long, default_value_t = 1_u8)]
    pub min_threads: u8,

    /// At what amount of threads to stop benchmarking.
    #[arg(long, default_value_t = 20_u8)]
    pub max_threads: u8,
}

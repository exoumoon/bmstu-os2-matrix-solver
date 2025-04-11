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
}

#![feature(portable_simd)]
#![allow(clippy::cast_precision_loss)]

use clap::Parser;
use color_eyre::eyre::Report;
use color_eyre::owo_colors::OwoColorize;
use sprs::io::{read_matrix_market, read_matrix_market_from_bufread};
use sprs::SparseMat;
use std::io::Cursor;

pub mod cli;

/// A 39x39 sparse matrix with 131 non-zero cells, used as fallback.
pub const FALLBACK_MATRIX_STR: &str = include_str!("../assets/mtx/bcsstk05_integer.mtx");

fn main() -> Result<(), Report> {
    color_eyre::install()?;

    let options = cli::Options::parse();
    let matrix = match options.matrix_path {
        Some(path) => read_matrix_market::<i32, usize, _>(path)?,
        None => read_matrix_market_from_bufread(&mut Cursor::new(FALLBACK_MATRIX_STR))?,
    };

    eprintln!("Loaded matrix: {}", SparseMatrixInfo::paramaters(&matrix));

    Ok(())
}

pub struct SparseMatrixInfo;

impl SparseMatrixInfo {
    pub fn paramaters<M: SparseMat>(matrix: &M) -> String {
        format!(
            "[{}×{}], {} nonzero cells",
            matrix.rows().blue().bold(),
            matrix.cols().blue().bold(),
            matrix.nnz().magenta().bold(),
        )
    }
}

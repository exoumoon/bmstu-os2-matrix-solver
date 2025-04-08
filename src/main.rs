#![feature(portable_simd)]
#![allow(clippy::cast_precision_loss)]

use clap::Parser;
use color_eyre::eyre::Report;
use color_eyre::owo_colors::OwoColorize;
use sprs::io::{read_matrix_market, read_matrix_market_from_bufread};
use sprs::SparseMat;
use std::io::Cursor;

pub mod cli;

/// 39x39 разреженная матрица, зашитая в программу. Используется по
/// умолчанию, если не указан путь до какого-либо .mtx файла.
pub const FALLBACK_MATRIX_STR: &str = include_str!("../assets/mtx/bcsstk05_integer.mtx");

fn main() -> Result<(), Report> {
    color_eyre::install()?;

    let options = cli::Options::parse();
    let matrix = match options.matrix_path {
        // Если путь передан, считываем из файла по тому пути. `sprs` не умеет читать
        // `pattern` матрицы, поэтому это обязательно должна быть матрица типа `integer`.
        Some(path) => read_matrix_market::<i32, usize, _>(path)?,

        // Если путь не передан, считываем матрицу из зашитой в программу.
        None => read_matrix_market_from_bufread(&mut Cursor::new(FALLBACK_MATRIX_STR))?,
    };

    // Матрица успешно считана, выведем информацию об её параметрах.
    eprintln!("Loaded matrix: {}", SparseMatrixInfo::paramaters(&matrix));

    Ok(())
}

// NOTE:
// Просто адаптер для отображения информации, не имеет отношения к решению.
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

#![feature(portable_simd)]
#![allow(
    clippy::cast_precision_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::suboptimal_flops,
    clippy::many_single_char_names
)]

use clap::Parser;
use color_eyre::eyre::Report;
use color_eyre::owo_colors::OwoColorize;
use sprs::io::{read_matrix_market, read_matrix_market_from_bufread};
use sprs::{CsMat, SparseMat};
use std::io::Cursor;

pub mod cli;

/// 39x39 разреженная матрица, зашитая в программу. Используется по
/// умолчанию, если не указан путь до какого-либо .mtx файла.
pub const FALLBACK_MATRIX_STR: &str = include_str!("../assets/mtx/3by3_integer.mtx");

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
    eprintln!("{:?}", matrix.to_csr::<usize>().to_dense());

    let vector = vec![1.0; matrix.rows()];
    let solution = bicgstab_preconditioned(&matrix.to_csr(), &vector, 10e-4, 1000).unwrap();
    eprintln!("Solution: {solution:.2?} [len: {}]", solution.len().red());

    Ok(())
}

fn jacobi_preconditioner(matrix: &CsMat<i32>) -> Vec<f64> {
    let num_columns = matrix.cols();
    let mut inverted = vec![0.0; num_columns];
    for (index, value) in inverted.iter_mut().enumerate() {
        if let Some(&cell_value) = matrix.get(index, index) {
            if cell_value != 0 {
                *value = 1.0 / f64::from(cell_value);
            }
        }
    }

    inverted
}

fn spmv(a: &CsMat<i32>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.rows()];
    for (r, row) in a.outer_iterator().enumerate() {
        for (c, cell) in row.iter() {
            y[r] += f64::from(*cell) * x[c];
        }
    }
    y
}

fn dot_product(vector_a: &[f64], vector_b: &[f64]) -> f64 {
    vector_a
        .iter()
        .zip(vector_b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

fn norm(v: &[f64]) -> f64 {
    dot_product(v, v).sqrt()
}

pub fn bicgstab_preconditioned(
    matrix: &CsMat<i32>,
    vector: &[f64],
    tolerance: f64,
    num_iterations: usize,
) -> Result<Vec<f64>, &'static str> {
    let n = matrix.cols();
    let mut x = vec![0.0; n];
    let m_inv = jacobi_preconditioner(matrix);

    let apply_preconditioner =
        |v: &[f64]| -> Vec<f64> { v.iter().zip(&m_inv).map(|(vi, mi)| vi * mi).collect() };

    let mut r = {
        let ax = spmv(matrix, &x);
        vector
            .iter()
            .zip(ax.iter())
            .map(|(bi, ai)| bi - ai)
            .collect::<Vec<_>>()
    };

    let r_tld = r.clone();
    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];

    let normb = norm(vector);
    if normb == 0.0 {
        return Ok(x);
    }

    for _ in 0..num_iterations {
        let rho_new = dot_product(&r_tld, &r);
        if rho_new.abs() < f64::EPSILON {
            return Err("Breakdown: rho ~ 0");
        }

        let beta = (rho_new / rho) * (alpha / omega);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        rho = rho_new;

        let p_hat = apply_preconditioner(&p);
        v = spmv(matrix, &p_hat);
        alpha = rho / dot_product(&r_tld, &v);
        let s: Vec<f64> = r
            .iter()
            .zip(v.iter())
            .map(|(ri, vi)| ri - alpha * vi)
            .collect();

        if norm(&s) < tolerance * normb {
            for i in 0..n {
                x[i] += alpha * p_hat[i];
            }
            return Ok(x);
        }

        let s_hat = apply_preconditioner(&s);
        let t = spmv(matrix, &s_hat);
        omega = dot_product(&t, &s) / dot_product(&t, &t);

        for i in 0..n {
            x[i] += alpha * p_hat[i] + omega * s_hat[i];
        }

        r = s
            .iter()
            .zip(t.iter())
            .map(|(si, ti)| si - omega * ti)
            .collect();

        if norm(&r) < tolerance * normb {
            return Ok(x);
        }

        if omega.abs() < f64::EPSILON {
            return Err("Breakdown: omega ~ 0");
        }
    }

    Err("BiCGSTAB did not converge within the maximum number of iterations")
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

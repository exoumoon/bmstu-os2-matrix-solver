#![feature(portable_simd)]
#![expect(clippy::redundant_clone)]
#![allow(
    clippy::cast_precision_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::suboptimal_flops,
    clippy::many_single_char_names
)]

// NOTE:
// ||Ax - b||         < e при dim < 10.000
// ||Ax - b|| / ||b|| < e при dim >= 10.000

use clap::Parser;
use color_eyre::eyre::Report;
use color_eyre::owo_colors::OwoColorize;
use sprs::io::{read_matrix_market, read_matrix_market_from_bufread};
use sprs::{CsMat, SparseMat};
use std::io::Cursor;
use std::simd::num::SimdFloat;
use std::simd::Simd;

pub mod benchmark;
pub mod cli;

pub const FALLBACK_MATRIX_STR: &str = include_str!("../assets/mtx/3by3_integer.mtx");

pub const FLOAT_TOLERANCE: f64 = 10e-4;
pub const MAX_BICGSTAB_ITERATIONS: usize = 1_000_000;
pub const MAX_RAYON_THREADS: usize = 100;

fn main() -> Result<(), Report> {
    color_eyre::install()?;

    let options = cli::Options::parse();
    let matrix = match options.matrix_path {
        // Если путь передан, считываем из файла по тому пути. `sprs` не умеет читать
        // `pattern` матрицы, поэтому это обязательно должна быть матрица типа `integer`.
        Some(path) => read_matrix_market::<f64, usize, _>(path)?,

        // Если путь не передан, считываем матрицу из зашитой в программу.
        None => read_matrix_market_from_bufread(&mut Cursor::new(FALLBACK_MATRIX_STR))?,
    };

    eprintln!("Loaded matrix: {}", SparseMatrixInfo::paramaters(&matrix));
    eprintln!("{:?}", matrix.to_csr::<usize>().to_dense());

    let vector = vec![1.0; matrix.rows()];

    let _ = benchmark::run_benchmark(
        &matrix.to_csr(),
        &vector,
        FLOAT_TOLERANCE,
        MAX_BICGSTAB_ITERATIONS,
        MAX_RAYON_THREADS,
    );

    Ok(())
}

fn jacobi_preconditioner(matrix: &CsMat<f64>) -> Vec<f64> {
    let num_columns = matrix.cols();
    let mut inverted = vec![0.0; num_columns];
    for (index, value) in inverted.iter_mut().enumerate() {
        if let Some(&cell_value) = matrix.get(index, index) {
            if cell_value != 0.0 {
                *value = 1.0 / cell_value;
            }
        }
    }

    inverted
}

// TODO: Parallelize (sparse matrix) * (regular vector).
fn sparse_matrix_mul_vector(a: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.rows()];
    for (r, row) in a.outer_iterator().enumerate() {
        for (c, cell) in row.iter() {
            y[r] += *cell * x[c];
        }
    }
    y
}

fn dot_product(vector_a: &[f64], vector_b: &[f64]) -> f64 {
    const LANES: usize = 8;
    type SimdType = Simd<f64, LANES>;

    let len = vector_a.len().min(vector_b.len());
    let chunks = len / LANES;

    let mut simd_sum = SimdType::splat(0.0);
    for i in 0..chunks {
        let a_chunk = SimdType::from_slice(&vector_a[i * LANES..][..LANES]);
        let b_chunk = SimdType::from_slice(&vector_b[i * LANES..][..LANES]);
        simd_sum += a_chunk * b_chunk;
    }

    let mut total = simd_sum.reduce_sum();

    for i in chunks * LANES..len {
        total += vector_a[i] * vector_b[i];
    }

    total
}

fn norm(vector: &[f64]) -> f64 {
    dot_product(vector, vector).sqrt()
}

pub fn bicgstab_preconditioned(
    matrix: &CsMat<f64>,
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
        let ax = sparse_matrix_mul_vector(matrix, &x);
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
        v = sparse_matrix_mul_vector(matrix, &p_hat);
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
        let t = sparse_matrix_mul_vector(matrix, &s_hat);
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

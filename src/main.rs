#![feature(portable_simd)]
#![expect(
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::redundant_clone,
    clippy::suboptimal_flops
)]

// NOTE: Verification:
// ||Ax - b||         < e при dim < 10.000
// ||Ax - b|| / ||b|| < e при dim >= 10.000

use clap::Parser;
use color_eyre::eyre::Report;
use nalgebra_sparse::{CsrMatrix, SparseEntry};
use rayon::prelude::*;
use std::simd::num::SimdFloat;
use std::simd::Simd;
use std::time::Instant;
use tracing::instrument;

pub mod benchmark;
pub mod cli;

pub const FALLBACK_MATRIX_STR: &str = include_str!("../assets/mtx/3by3_integer.mtx");

pub const FLOAT_TOLERANCE: f64 = 10e-4;
pub const MAX_BICGSTAB_ITERATIONS: usize = 1_000_000;
pub const MAX_RAYON_THREADS: usize = 100;

fn main() -> Result<(), Report> {
    color_eyre::install()?;
    install_tracing()?;

    let options = cli::Options::parse();
    let coo_matrix = match options.matrix_path {
        // Если путь передан, считываем из файла по тому пути. `sprs` не умеет читать
        // `pattern` матрицы, поэтому это обязательно должна быть матрица типа `integer`.
        Some(path) => nalgebra_sparse::io::load_coo_from_matrix_market_file(path)?,

        // Если путь не передан, считываем матрицу из зашитой в программу.
        None => nalgebra_sparse::io::load_coo_from_matrix_market_str(FALLBACK_MATRIX_STR)?,
    };

    let csr_matrix = CsrMatrix::from(&coo_matrix);
    let vector = vec![1.0; csr_matrix.nrows()];

    let _ = benchmark::run_benchmark(
        &csr_matrix,
        &vector,
        FLOAT_TOLERANCE,
        MAX_BICGSTAB_ITERATIONS,
        MAX_RAYON_THREADS,
    );

    Ok(())
}

fn jacobi_preconditioner(matrix: &CsrMatrix<f64>) -> Vec<f64> {
    let mut inverted_vector = vec![0.0; matrix.ncols()];
    for (index, value) in inverted_vector.iter_mut().enumerate() {
        if let Some(sparse_entry) = matrix.get_entry(index, index) {
            match sparse_entry {
                SparseEntry::Zero => {}
                SparseEntry::NonZero(cell_value) => {
                    if *cell_value != 0.0 {
                        *value = 1.0 / cell_value;
                    }
                }
            }
        }
    }

    inverted_vector
}

/// Parallelized multiplication of a sparse matrix by a dense vector.
#[instrument(
    skip_all,
    fields(
        matrix.rows = matrix.nrows(),
        matrix.columns = matrix.ncols(),
        matrix.nonzero_cells = matrix.nnz(),
        vector.len = vector.len(),
        timings.multiplication,
        timings.restoration,
    ),
)]
fn parallelized_spmv(matrix: &CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let start = Instant::now();
    let mut result = matrix
        .row_iter()
        .enumerate()
        .par_bridge() /* <-- NOTE: Код после этого выполняется параллельно */
        .map(|(row_index, row)| {
            let sum = row
                .values()
                .iter()
                .zip(row.col_indices())
                .map(|(row_value, index)| row_value * vector[*index])
                .sum();
            (row_index, sum)
        })
        .collect::<Vec<_>>(); /* WARN: Необходимо восстановить порядок строк! */
    tracing::debug!(duration = ?start.elapsed(), "Finished multiplying sparse matrix by vector");
    let before_sorting = Instant::now();
    result.sort_unstable_by_key(|(row_index, _)| *row_index);
    tracing::debug!(duration = ?before_sorting.elapsed(), "Restored row order");

    result.into_iter().map(|(_, value)| value).collect()
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
    matrix: &CsrMatrix<f64>,
    vector: &[f64],
    tolerance: f64,
    num_iterations: usize,
) -> Result<Vec<f64>, &'static str> {
    let n = matrix.ncols();
    let mut x = vec![0.0; n];
    let m_inv = jacobi_preconditioner(matrix);

    let apply_preconditioner =
        |v: &[f64]| -> Vec<f64> { v.iter().zip(&m_inv).map(|(vi, mi)| vi * mi).collect() };

    let mut r = {
        let ax = parallelized_spmv(matrix, &x);
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
        v = parallelized_spmv(matrix, &p_hat);
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
        let t = parallelized_spmv(matrix, &s_hat);
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

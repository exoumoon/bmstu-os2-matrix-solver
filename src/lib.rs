#![feature(portable_simd)]
#![expect(
    clippy::cast_precision_loss,
    clippy::redundant_clone,
    clippy::suboptimal_flops
)]

use nalgebra_sparse::{CsrMatrix, SparseEntry};
use rayon::prelude::*;
use std::simd::num::SimdFloat;
use std::simd::Simd;
use std::time::{Duration, Instant};
use tracing::instrument;

#[derive(Debug)]
#[must_use]
pub struct Spmv;

impl Spmv {
    /// Serial multiplication of a sparse matrix by a dense vector.
    #[instrument(
        name = "serial_spmv"
        skip_all,
        fields(
            matrix.rows = matrix.nrows(),
            matrix.columns = matrix.ncols(),
            matrix.nonzero_cells = matrix.nnz(),
            vector.len = vector.len(),
        ),
    )]
    pub fn serial(&self, matrix: &CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
        let start = Instant::now();
        let result: Vec<f64> = matrix
            .row_iter()
            .map(|row| {
                row.values()
                    .iter()
                    .zip(row.col_indices())
                    .map(|(row_value, index)| row_value * vector[*index])
                    .sum::<f64>()
            })
            .collect();
        tracing::debug!(op_duration = ?start.elapsed(), "Multiplied sparse matrix by vector");
        result
    }

    /// Parallelized multiplication of a sparse matrix by a dense vector.
    #[instrument(
        name = "parallelized_spmv"
        skip_all,
        fields(
            matrix.rows = matrix.nrows(),
            matrix.columns = matrix.ncols(),
            matrix.nonzero_cells = matrix.nnz(),
            vector.len = vector.len(),
            threads.num = rayon::current_num_threads(),
            threads.max = rayon::max_num_threads(),
        ),
    )]
    pub fn parallelized(&self, matrix: &CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
        let start = Instant::now();
        let result: Vec<f64> = matrix
            .row_iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|row| {
                /* NOTE: Код в этом блоке выполняется параллельно */
                row.values()
                    .iter()
                    .zip(row.col_indices())
                    .map(|(row_value, index)| row_value * vector[*index])
                    .sum::<f64>()
            })
            .collect();
        tracing::debug!(op_duration = ?start.elapsed(), "Multiplied sparse matrix by vector");
        result
    }
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

#[instrument(skip(matrix, vector))]
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
        let ax = Spmv.parallelized(matrix, &x);
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

    let start = Instant::now();
    let mut last_checkpoint = start;
    for iter in 0..num_iterations {
        {
            if iter > 0 && iter % (num_iterations / 1000) == 0 {
                let completion = iter as f64 / num_iterations as f64;
                let since_checkpoint = last_checkpoint.elapsed();
                let estimated_completion_secs = start.elapsed().as_secs_f64() / completion;
                let estimated_remaining =
                    Duration::from_secs_f64(estimated_completion_secs) - start.elapsed();
                tracing::info!(
                    completion,
                    ?since_checkpoint,
                    ?estimated_remaining,
                    ?estimated_completion_secs,
                    "Completed {iter} of {num_iterations} iterations"
                );
                last_checkpoint = Instant::now();
            }
        }
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
        v = Spmv.parallelized(matrix, &p_hat);
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
        let t = Spmv.parallelized(matrix, &s_hat);
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

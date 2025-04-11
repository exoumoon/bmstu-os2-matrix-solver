#![feature(portable_simd)]
#![expect(
    clippy::cast_precision_loss,
    clippy::redundant_clone,
    clippy::suboptimal_flops
)]

use nalgebra_sparse::CsrMatrix;
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
    #[must_use]
    pub fn serial(&self, matrix: &CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
        let result: Vec<f64> = matrix
            .row_iter()
            .map(|row| {
                /* NOTE: Код в этом блоке выполняется параллельно */
                row.values()
                    .iter()
                    .zip(row.col_indices())
                    .map(|(row_value, index)| row_value * vector[*index])
                    .sum::<f64>()
            })
            .collect();
        result
    }

    /// Parallelized multiplication of a sparse matrix by a dense vector.
    #[must_use]
    pub fn parallelized(&self, matrix: &CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
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
        result
    }
}

#[must_use]
pub fn jacobi_preconditioner(matrix: &CsrMatrix<f64>) -> Vec<f64> {
    let mut result = vec![0.0; matrix.nrows()];
    for (row_index, row) in matrix.row_iter().enumerate() {
        let diagonal_index = row
            .col_indices()
            .iter()
            .position(|&col_index| col_index == row_index);
        if let Some(index) = diagonal_index {
            let value = row.values()[index];
            if value != 0.0 {
                result[row_index] = 1.0 / value;
            }
        }
    }

    result
}

#[must_use]
pub fn dot_product_simd(vector_a: &[f64], vector_b: &[f64]) -> f64 {
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

#[must_use]
pub fn dot_product_scalar(vector_a: &[f64], vector_b: &[f64]) -> f64 {
    vector_a
        .iter()
        .zip(vector_b.iter())
        .map(|(a, b)| a * b)
        .sum()
}

fn norm(vector: &[f64]) -> f64 {
    dot_product_simd(vector, vector).sqrt()
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
                tracing::debug!(
                    completion,
                    ?since_checkpoint,
                    ?estimated_remaining,
                    ?estimated_completion_secs,
                    "Completed {iter} of {num_iterations} iterations"
                );
                last_checkpoint = Instant::now();
            }
        }
        let rho_new = dot_product_simd(&r_tld, &r);
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
        alpha = rho / dot_product_simd(&r_tld, &v);
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
        omega = dot_product_simd(&t, &s) / dot_product_simd(&t, &t);

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

pub mod io {
    use nalgebra_sparse::io::{self, MatrixMarketError};
    use std::path::Path;

    #[expect(clippy::missing_errors_doc)]
    pub fn load_vector_from_matrix_market_file<P>(path: P) -> Result<Vec<f64>, MatrixMarketError>
    where
        P: AsRef<Path>,
    {
        let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(path)?;
        let mut vector = vec![0.0; coo_matrix.nrows()];

        for (index, value) in coo_matrix.row_indices().iter().zip(coo_matrix.values()) {
            vector[*index] = *value;
        }

        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use crate::Spmv;
    use nalgebra::DVector;
    use nalgebra_sparse::{io, CsrMatrix};

    #[rstest::rstest]
    #[case("assets/mtx/e20r0000_rhs1.mtx")]
    #[case("assets/mtx/e40r0000_rhs1.mtx")]
    #[case("assets/mtx/fidap011_rhs1.mtx")]
    fn load_rhs_vector(#[case] path: &str) {
        let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(path).unwrap();
        let mut vector = vec![0.0; coo_matrix.nrows()];

        for (index, value) in coo_matrix.row_indices().iter().zip(coo_matrix.values()) {
            vector[*index] = *value;
        }

        eprintln!("vector: {vector:?}");
        assert_eq!(vector.len(), coo_matrix.nrows());
    }

    #[rstest::rstest]
    // NOTE: These take way too long.
    // #[case("assets/mtx/e20r0000.mtx", "assets/mtx/e20r0000_rhs1.mtx")]
    // #[case("assets/mtx/e40r0000.mtx", "assets/mtx/e40r0000_rhs1.mtx")]
    #[case("assets/mtx/fidap011.mtx", "assets/mtx/fidap011_rhs1.mtx")]
    fn bicgstab(#[case] matrix_path: &str, #[case] rhs_path: &str) {
        use std::f64::consts::E;

        let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(matrix_path).unwrap();
        let a = CsrMatrix::from(&coo_matrix);
        let b = crate::io::load_vector_from_matrix_market_file(rhs_path).unwrap();
        let x = crate::bicgstab_preconditioned(&a, &b, 10e-4, 1_000_000).unwrap();

        let ax = Spmv.parallelized(&a, &x);
        let dv_b = DVector::from_vec(b);
        let dv_ax = DVector::from_vec(ax);

        let ax_minus_b = dv_ax - dv_b.clone();
        let ax_minus_b_norm = ax_minus_b.norm();
        let b_norm = dv_b.norm();
        dbg!(a.nrows(), ax_minus_b_norm, b_norm, ax_minus_b_norm / b_norm);

        match a.nrows() {
            ..10_000 => assert!(ax_minus_b_norm <= E),
            10_000.. => assert!(ax_minus_b_norm / b_norm <= E),
        }
    }
}

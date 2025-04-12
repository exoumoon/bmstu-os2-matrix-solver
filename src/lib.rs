#![feature(portable_simd, slice_pattern)]
#![allow(
    clippy::cast_precision_loss,
    clippy::redundant_clone,
    clippy::suboptimal_flops,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names
)]

use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::*;
use std::simd::num::SimdFloat;
use std::simd::Simd;

pub mod benchmark;
pub mod io;

pub const FLOAT_TOLERANCE: f64 = 10e-6;
pub const MAX_ITERATIONS: usize = 1_000_000;
pub const MAX_THREADS: usize = 20;

#[derive(Debug)]
#[must_use]
pub struct Spmv;

impl Spmv {
    /// Serial multiplication of a sparse matrix by a dense vector.
    #[must_use]
    pub fn serial(&self, matrix: &CsrMatrix<f64>, vector: &DVector<f64>) -> DVector<f64> {
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
        result.into()
    }

    /// Parallelized multiplication of a sparse matrix by a dense vector.
    #[must_use]
    pub fn parallelized(&self, matrix: &CsrMatrix<f64>, vector: &DVector<f64>) -> DVector<f64> {
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
        result.into()
    }
}

#[must_use]
pub fn jacobi_preconditioner(matrix: &CsrMatrix<f64>) -> DVector<f64> {
    let mut result = vec![1.0; matrix.nrows()];
    for (row_index, row) in matrix.row_iter().enumerate() {
        let diagonal_index = row
            .col_indices()
            .iter()
            .position(|&col_index| col_index == row_index);
        if let Some(index) = diagonal_index {
            let value = row.values()[index];
            if value.abs() <= f64::EPSILON {
                result[row_index] = 1.0 / value;
            }
        }
    }

    result.into()
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

#[derive(Debug, thiserror::Error)]
pub enum BicgstabError {
    #[error("Failed to find a solution within the iteration limit")]
    NoConvergence,
    #[error("`rho` turned near-zero")]
    BiOrthogonalityLost,
    #[error("`omega` turned near-zero")]
    OmegaClamped,
}

pub fn bicgstab_preconditioned(
    matrix: &CsrMatrix<f64>,
    b: &DVector<f64>,
    float_tolerance: f64,
    max_iterations: usize,
) -> Result<DVector<f64>, BicgstabError> {
    let dim = matrix.ncols();

    let m_inv: DVector<f64> = jacobi_preconditioner(matrix);
    let apply_preconditioner = |v: &DVector<f64>| -> DVector<f64> { v.component_mul(&m_inv) };

    let mut guess: DVector<f64> = DVector::from_vec(vec![0.0; dim]);
    let mut residual: DVector<f64> = b - Spmv.parallelized(matrix, &guess);
    let r_tilda = residual.clone();
    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut v: DVector<f64> = vec![0.0; dim].into();
    let mut p: DVector<f64> = vec![0.0; dim].into();

    let normb = b.norm();
    if normb.abs() <= f64::EPSILON {
        return Ok(guess);
    }

    for _ in 0..max_iterations {
        let rho_new = dot_product_simd(r_tilda.as_slice(), residual.as_slice());
        if rho_new.abs() < f64::EPSILON {
            return Err(BicgstabError::BiOrthogonalityLost);
        }

        let beta = (rho_new / rho) * (alpha / omega);
        for i in 0..dim {
            p[i] = residual[i] + beta * (p[i] - omega * v[i]);
        }

        rho = rho_new;

        let p_hat = apply_preconditioner(&p);
        v = Spmv.parallelized(matrix, &p_hat);
        alpha = rho / dot_product_simd(r_tilda.as_slice(), v.as_slice());
        let s: DVector<f64> = residual - (alpha * v.clone());

        if s.norm() < float_tolerance * normb {
            for i in 0..dim {
                guess[i] += alpha * p_hat[i];
            }
            return Ok(guess);
        }

        let s_hat = apply_preconditioner(&s);
        let t = Spmv.parallelized(matrix, &s_hat);
        omega = dot_product_simd(t.as_slice(), s.as_slice())
            / dot_product_simd(t.as_slice(), t.as_slice());

        for i in 0..dim {
            guess[i] += alpha * p_hat[i] + omega * s_hat[i];
        }

        // PERF: SIMD-accelerated in-place `residual = s - omega * t`.
        residual = s;
        residual.axpy(-omega, &t, 1.0);

        if residual.norm() < float_tolerance * normb {
            return Ok(guess);
        }

        if omega.abs() < f64::EPSILON {
            return Err(BicgstabError::OmegaClamped);
        }
    }

    Err(BicgstabError::NoConvergence)
}

#[cfg(test)]
mod tests {
    use crate::{Spmv, FLOAT_TOLERANCE, MAX_ITERATIONS};
    use color_eyre::eyre::Report;
    use nalgebra_sparse::{io, CsrMatrix};

    #[rstest::rstest]
    #[case::e20r0000("assets/mtx/e20r0000_rhs1.mtx")]
    #[case::e40r0000("assets/mtx/e40r0000_rhs1.mtx")]
    #[case::fidap011("assets/mtx/fidap011_rhs1.mtx")]
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
    // #[case::fidapm37("assets/mtx/fidapm37.mtx", "assets/mtx/fidapm37_rhs1.mtx")]
    #[case::fidap011("assets/mtx/fidap011.mtx", "assets/mtx/fidap011_rhs1.mtx")]
    fn bicgstab(#[case] matrix_path: &str, #[case] rhs_path: &str) -> Result<(), Report> {
        use std::f64::consts::E;
        let _ = color_eyre::install();

        let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(matrix_path)?;
        let a = CsrMatrix::from(&coo_matrix);
        let b = crate::io::load_vector_from_matrix_market_file(rhs_path)?;
        let x = crate::bicgstab_preconditioned(&a, &b, FLOAT_TOLERANCE, MAX_ITERATIONS)?;
        let ax = Spmv.parallelized(&a, &x);

        let ax_minus_b_norm = (ax - b.clone()).norm();
        let b_norm = b.norm();
        dbg!(a.nrows(), ax_minus_b_norm, b_norm, ax_minus_b_norm / b_norm);

        match a.nrows() {
            // NOTE: Verification:
            // ||Ax - b||         < E, при dim < 10.000
            // ||Ax - b|| / ||b|| < E, при dim >= 10.000
            ..10_000 => assert!(ax_minus_b_norm <= E),
            10_000.. => assert!(ax_minus_b_norm / b_norm <= E),
        }

        Ok(())
    }
}

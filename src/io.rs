use nalgebra::DVector;
use nalgebra_sparse::io::{self, MatrixMarketError};
use std::path::Path;

pub fn load_vector_from_matrix_market_file<P>(path: P) -> Result<DVector<f64>, MatrixMarketError>
where
    P: AsRef<Path>,
{
    let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(path)?;
    let mut vector = vec![0.0; coo_matrix.nrows()];

    for (index, value) in coo_matrix.row_indices().iter().zip(coo_matrix.values()) {
        vector[*index] = *value;
    }

    Ok(vector.into())
}

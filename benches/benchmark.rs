use bmstu_os2_matrix_solver::{dot_product_scalar, dot_product_simd, jacobi_preconditioner, Spmv};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra_sparse::{io, CsrMatrix};
use rand::Rng;
use rayon::ThreadPoolBuilder;
use std::time::Duration;

const MTX_PATHS: [&str; 3] = [
    "assets/mtx/e20r0000.mtx",
    "assets/mtx/e40r0000.mtx",
    "assets/mtx/fidap011.mtx",
];

fn spmv_benchmarks(criterion: &mut Criterion) {
    let mut rng = rand::rng();
    let mut group = criterion.benchmark_group("spmv");
    group
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(5));

    for mtx_path in MTX_PATHS {
        let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(mtx_path).unwrap();
        let csr_matrix = CsrMatrix::from(&coo_matrix);
        let random_vector = (0..csr_matrix.nrows())
            .map(|_| rng.random::<f64>())
            .collect::<Vec<_>>();

        let id = format!("serial-{mtx_path}");
        group.bench_function(&id, |b| {
            b.iter(|| Spmv.serial(&csr_matrix, &random_vector));
        });

        for thread_count in [4, 8, 20] {
            let id = format!("parallel-{thread_count}threads-{mtx_path}");
            group.bench_function(&id, |b| {
                let thread_pool = ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .unwrap();
                thread_pool.install(|| {
                    b.iter(|| Spmv.parallelized(&csr_matrix, &random_vector));
                });
            });
        }
    }
}

fn preconditioner_benchmarks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("preconditioners");
    group
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for mtx_path in MTX_PATHS {
        let coo_matrix = io::load_coo_from_matrix_market_file::<f64, _>(mtx_path).unwrap();
        let csr_matrix = CsrMatrix::from(&coo_matrix);

        let id = format!("jacobi-{mtx_path}");
        group.bench_function(&id, |b| {
            b.iter(|| jacobi_preconditioner(&csr_matrix));
        });
    }
}

fn dotproduct_benchmarks(criterion: &mut Criterion) {
    let mut rng = rand::rng();
    let mut group = criterion.benchmark_group("dotproduct");
    group
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(3));

    for vector_length in (16..=20_u32).map(|power| 2_usize.pow(power)) {
        let vec_a = (0..vector_length)
            .map(|_| rng.random::<f64>())
            .collect::<Vec<_>>();
        let vec_b = (0..vector_length)
            .map(|_| rng.random::<f64>())
            .collect::<Vec<_>>();

        let id = format!("scalar-len{vector_length}");
        group.bench_function(&id, |b| {
            b.iter(|| {
                let _ = black_box(dot_product_scalar(&vec_a, &vec_b));
            });
        });

        let id = format!("simd-len{vector_length}");
        group.bench_function(&id, |b| {
            b.iter(|| {
                let _ = black_box(dot_product_simd(&vec_a, &vec_b));
            });
        });
    }
}

criterion_group!(
    benches,
    spmv_benchmarks,
    preconditioner_benchmarks,
    dotproduct_benchmarks
);
criterion_main!(benches);

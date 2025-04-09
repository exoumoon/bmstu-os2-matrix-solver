use super::bicgstab_preconditioned;
use rayon::ThreadPoolBuilder;
use sprs::CsMat;
use std::time::Instant;

#[derive(Debug, Clone)]
#[must_use]
pub struct BenchmarkResult {
    pub p: usize,
    pub t1: f64,
    pub tp: f64,
    pub speedup: f64,
    pub efficiency: f64,
    pub alpha: f64,
}

fn amdahl_alpha(p: f64, sp: f64) -> f64 {
    if (p - 1.0).abs() < f64::EPSILON {
        return 0.0;
    }
    (1.0 - 1.0 / p) / (1.0 / sp - 1.0 / p)
}

#[must_use]
pub fn run_benchmark(
    matrix: &CsMat<f64>,
    b: &[f64],
    tolerance: f64,
    iterations: usize,
    max_threads: usize,
) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    let mut t1 = 0.0;

    for num_threads in 1..=max_threads {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to build thread pool");

        let start = Instant::now();
        pool.install(|| {
            bicgstab_preconditioned(matrix, b, tolerance, iterations).unwrap();
        });
        let elapsed = start.elapsed().as_secs_f64();

        if num_threads == 1 {
            t1 = elapsed;
        }

        let sp = t1 / elapsed;
        let ep = sp / num_threads as f64;
        let alpha = amdahl_alpha(num_threads as f64, sp);

        let result = BenchmarkResult {
            p: num_threads,
            t1,
            tp: elapsed,
            speedup: sp,
            efficiency: ep,
            alpha,
        };

        results.push(result.clone());

        println!(
            "p = {:>3} | t₁ = {:>7.4}s | tₚ = {:>7.4}s | Sₚ = {:>5.2} | Eₚ = {:>5.2}% | α = {:>5.3}",
            result.p,
            result.t1,
            result.tp,
            result.speedup,
            result.efficiency * 100.0,
            result.alpha
        );
    }

    results
}

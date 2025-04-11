use bmstu_os2_matrix_solver::bicgstab_preconditioned;
use color_eyre::owo_colors::OwoColorize;
use itertools::Itertools;
use nalgebra_sparse::CsrMatrix;
use plotly::common::Mode;
use plotly::layout::{GridPattern, LayoutGrid};
use plotly::{Layout, Plot, Scatter};
use rayon::ThreadPoolBuilder;
use std::fmt;
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

#[derive(Debug, Clone)]
#[must_use]
pub struct BenchmarkResults {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkResults {
    #[must_use]
    pub fn create_plots(&self) -> Plot {
        let thread_points = self.results.iter().map(|r| r.p).collect_vec();
        let t1_points = self.results.iter().map(|r| r.t1).collect_vec();
        let time_points = self.results.iter().map(|r| r.tp).collect_vec();
        let speedup_points = self.results.iter().map(|r| r.speedup).collect_vec();
        let efficiency_points = self.results.iter().map(|r| r.efficiency).collect_vec();
        let alpha_points = self.results.iter().map(|r| r.alpha).collect_vec();

        let t1_trace = Scatter::new(thread_points.clone(), t1_points)
            .name("Время с одним потоком (t₁)")
            .mode(Mode::LinesMarkersText);
        let time_trace = Scatter::new(thread_points.clone(), time_points)
            .name("Время (tₚ)")
            .mode(Mode::LinesMarkersText);
        let speedup_trace = Scatter::new(thread_points.clone(), speedup_points)
            .name("Ускорение (Sₚ)")
            .x_axis("x2")
            .y_axis("y2")
            .mode(Mode::LinesMarkersText);
        let efficiency_trace = Scatter::new(thread_points.clone(), efficiency_points)
            .name("Эффективность (Eₚ)")
            .x_axis("x3")
            .y_axis("y3")
            .mode(Mode::LinesMarkersText);
        let alpha_trace = Scatter::new(thread_points, alpha_points)
            .name("Альфа (α)")
            .x_axis("x4")
            .y_axis("y4")
            .mode(Mode::LinesMarkersText);

        let mut plot = Plot::new();
        plot.add_trace(t1_trace);
        plot.add_trace(time_trace);
        plot.add_trace(speedup_trace);
        plot.add_trace(efficiency_trace);
        plot.add_trace(alpha_trace);

        let layout_grid = LayoutGrid::new()
            .rows(2)
            .columns(2)
            .pattern(GridPattern::Independent);
        plot.set_layout(Layout::new().grid(layout_grid));
        plot
    }
}

fn amdahl_alpha(p: f64, sp: f64) -> f64 {
    if (p - 1.0).abs() < f64::EPSILON {
        return 0.0;
    }
    (1.0 - 1.0 / p) / (1.0 / sp - 1.0 / p)
}

pub fn run_benchmark(
    matrix: &CsrMatrix<f64>,
    b: &[f64],
    tolerance: f64,
    iterations: usize,
    max_threads: usize,
) -> BenchmarkResults {
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
        println!("{result}");
    }

    BenchmarkResults { results }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,
            "p = {:>3} | t₁ = {:>7.4}s | tₚ = {:>7.4}s | Sₚ = {:>5.2} | Eₚ = {:>6.2}% | α = {:>5.3}",
            self.p.bold(),
            self.t1.red().bold(),
            self.tp.yellow().bold(),
            self.speedup.bright_yellow().bold(),
            (self.efficiency * 100.0).bright_red().bold(),
            self.alpha.bright_magenta().bold(),
        )?;
        Ok(())
    }
}

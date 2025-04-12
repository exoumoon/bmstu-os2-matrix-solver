use crate::bicgstab_preconditioned;
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
    pub num_threads: usize,
    pub time_serial: f64,
    pub time_parallel: f64,
    pub speedup: f64,
    pub efficiency: f64,
    pub serial_op_share: f64,
}

#[derive(Debug, Clone)]
#[must_use]
pub struct BenchmarkResults {
    results: Vec<BenchmarkResult>,
}

impl From<Vec<BenchmarkResult>> for BenchmarkResults {
    fn from(results: Vec<BenchmarkResult>) -> Self {
        Self { results }
    }
}

impl BenchmarkResults {
    #[must_use]
    pub fn create_plots(&self) -> Plot {
        let thread_points = self.results.iter().map(|r| r.num_threads).collect_vec();
        let t1_points = self.results.iter().map(|r| r.time_serial).collect_vec();
        let time_points = self.results.iter().map(|r| r.time_parallel).collect_vec();
        let speedup_points = self.results.iter().map(|r| r.speedup).collect_vec();
        let efficiency_points = self.results.iter().map(|r| r.efficiency).collect_vec();
        let alpha_points = self.results.iter().map(|r| r.serial_op_share).collect_vec();

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
            .name("Доля последовательных вычислений (α)")
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

#[must_use]
pub fn amdahl_alpha(num_threads: usize, speedup: f64) -> f64 {
    if num_threads <= 1 || speedup <= 0.0 || !speedup.is_finite() {
        return 1.0;
    }

    let denominator = num_threads as f64 - 1.0;
    if denominator.abs() < f64::EPSILON {
        return 1.0;
    }

    let alpha = (num_threads as f64 / speedup - 1.0) / denominator;
    alpha.clamp(0.0, 1.0)
}

pub fn run_benchmark(
    matrix: &CsrMatrix<f64>,
    b: &[f64],
    tolerance: f64,
    max_iterations: usize,
    max_threads: usize,
) -> Result<BenchmarkResults, color_eyre::eyre::Report> {
    let mut results = Vec::new();
    let mut time_serial = 0.0;

    for num_threads in 1..=max_threads {
        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build()?;

        let start = Instant::now();
        let solution =
            pool.install(|| bicgstab_preconditioned(matrix, b, tolerance, max_iterations));
        let elapsed_seconds = start.elapsed().as_secs_f64();
        if let Err(error) = solution {
            tracing::error!(?num_threads, ?error, "BiCGSTAB failed");
            continue;
        }

        if num_threads == 1 {
            time_serial = elapsed_seconds;
        }

        let speedup = time_serial / elapsed_seconds;
        let efficiency = speedup / num_threads as f64;
        let serial_op_share = amdahl_alpha(num_threads, speedup);

        let result = BenchmarkResult {
            num_threads,
            time_serial,
            time_parallel: elapsed_seconds,
            speedup,
            efficiency,
            serial_op_share,
        };
        println!("{result}");
        results.push(result);
    }

    Ok(BenchmarkResults::from(results))
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,
            "P = {p:>3} | t₁ = {t1:>7.4}s | tₚ = {tp:>7.4}s | Sₚ = {sp:>5.2} | Eₚ = {e:>6.2}% | α = {a:>5.3}",
            p = self.num_threads.bold(),
            t1 = self.time_serial.red().bold(),
            tp = self.time_parallel.yellow().bold(),
            sp = self.speedup.bright_yellow().bold(),
            e = (self.efficiency * 100.0).bright_red().bold(),
            a = self.serial_op_share.bright_magenta().bold(),
        )?;
        Ok(())
    }
}

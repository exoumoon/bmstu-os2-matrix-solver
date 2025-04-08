use color_eyre::eyre::Report;
use sprs::io::read_matrix_market;

fn main() -> Result<(), Report> {
    color_eyre::install()?;

    let matrix = read_matrix_market::<f64, usize, &str>("assets/mtx/bcsstk05.mtx")?;
    dbg!(&matrix);

    Ok(())
}

// fn bicgstab_preconditioned(
//     a: &CsMat<f64>,
//     b: &DVector<f64>,
//     m_inv: &dyn Fn(&DVector<f64>) -> DVector<f64>, /* Предобуславливатель */
//     tol: f64,
//     max_iter: usize,
// ) -> DVector<f64> {
//     let n = b.len();
//     let mut x = DVector::zeros(n);
//     let mut r = b - &a * &x;
//     let mut r_hat = r.clone();
//     let mut rho_old = 1.0;
//     let mut alpha = 1.0;
//     let mut omega = 1.0;
//     let mut v = DVector::zeros(n);
//     let mut p = DVector::zeros(n);

//     for _ in 0..max_iter {
//         let rho_new = r_hat.dot(&r);
//         if rho_new.abs() < 1e-20 {
//             break;
//         }

//         let beta = (rho_new / rho_old) * (alpha / omega);
//         p = &r + beta * (&p - omega * &v);

//         // Применяем предобуславливатель
//         let y = m_inv(&p);
//         v = &a * &y;
//         alpha = rho_new / r_hat.dot(&v);
//         let s = &r - alpha * &v;

//         if s.norm() < tol {
//             x += alpha * &y;
//             break;
//         }

//         // Применяем предобуславливатель к s
//         let z = m_inv(&s);
//         let t = &a * &z;
//         omega = t.dot(&s) / t.dot(&t);
//         x += alpha * &y + omega * &z;
//         r = &s - omega * &t;

//         if r.norm() < tol {
//             break;
//         }

//         rho_old = rho_new;
//     }

//     x
// }

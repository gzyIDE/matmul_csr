use ndarray::*;
use ndarray_linalg::*;
use rand::Rng;
use std::mem;

mod csr;
mod dense;
use csr::*;
use dense::*;

fn main() {
    let col = 100;
    let row = 100;
    let nnz_rate = 0.1;
    let mut rng = rand::thread_rng();

    // Matrix/Vector initialization
    let mat_a: Array2<f64> = random((row, col));
    let spmat_a = mat_a.mapv(|f| if rng.gen::<f64>() < nnz_rate {f} else {0.0});

    // ndarray-linalg function
    let vec_x: Array1<f64> = random(col);

    // DIY functions
    let mut vec_y: Array1<f64> = Array::zeros(row);
    mv_dense(&spmat_a, &vec_x, &mut vec_y);

    // Dense -> Compress Sparse Row
    let mut spmat_a_csr : CSR = CSR::new();
    d2s_csr(&spmat_a, &mut spmat_a_csr);

    // Compress Sparse Row -> Dense
    let mut spmat_b : Array2<f64> = Array::zeros((row, col));
    s2d_csr(&mut spmat_b, &spmat_a_csr);

    // Matrix Operation
    let mut vec_z: Array1<f64> = Array::zeros(row);
    SpMV_csr(&spmat_a_csr, &vec_x, &mut vec_z);
 
    println!("Dense -> Sparse -> Dense conversion check");
    if spmat_a != spmat_b {
        println!("  check failed");
    } else {
        println!("  check passed");
    }

    println!("Sparse matrix multiplication check");
    if vec_z != vec_y {
        println!("  check failed");
    } else {
        println!("  check passed");
    }
}

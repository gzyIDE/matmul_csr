use ndarray::*;
use ndarray_linalg::*;
use rand::Rng;
use std::mem;

mod csr;
mod dense;
use csr::*;
use dense::*;

fn main() {
    let col = 3;
    let row = 3;
    let nnz_rate = 1.0;
    let mut rng = rand::thread_rng();

    //***** Matrix Vector Multiply test
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

    // Answer
    let ans_vec_y = spmat_a.dot(&vec_x);

    // result checks
    for (i, (a, b)) in vec_y.iter().zip(ans_vec_y.iter()).enumerate() {
        // Compare result considering rounding error.
        // Probably "dot" of ndarray_linalg uses Fused Multiply and add (FMA),
        // which causes precision difference from separeted muliply and add.
        if (a - b).abs() > 0.000001 {
            println!("{}: fail", i);
        }
    }

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


    //***** Matrix Matrix Multiply test
    let mat_b: Array2<f64> = random((col, row));
    let spmat_b = mat_a.mapv(|f| if rng.gen::<f64>() < nnz_rate {f} else {0.0});

    let mut mat_z :Array2<f64> = Array::zeros((row, row));

    let ans_mat_z = spmat_a.dot(&spmat_b);

    mm_dense(&spmat_a, &spmat_b, &mut mat_z);

    let mut mat_z_nsa :Array2<f64> = Array::zeros((row, row));
    mm_systolic_nsa(&spmat_a, &spmat_b, &mut mat_z_nsa);

    // result checks
    let mut mm_error = false;
    println!("Matrix multiply check");
    for (i, (a, b)) in mat_z.iter().zip(ans_mat_z.iter()).enumerate() {
        // Compare result considering rounding error.
        if (a - b).abs() > 0.000001 {
            mm_error = true;
        }
    }
    if ( mm_error ) {
        println!("  check failed");
    } else {
        println!("  check passed");
    }

    // result checks (matrix multiply vs systolic)
    let mut sa_error = false;
    println!("Systolic matrix multiply check");
    for (i, (a, b)) in mat_z.iter().zip(mat_z_nsa.iter()).enumerate() {
        // Compare result considering rounding error.
        if (a - b).abs() > 0.000001 {
            sa_error = true;
        }
    }
    if ( sa_error ) {
        println!("  check failed");
    } else {
        println!("  check passed");
    }
}

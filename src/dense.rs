use ndarray::*;
use ndarray_linalg::*;

pub fn mv_dense(mat_a: &Array2<f64>, vec_x: &Array1<f64>, vec_y: &mut Array1<f64>) {
    let shape_a = mat_a.shape();
    let shape_x = vec_x.shape();
    let shape_y = vec_y.shape();
    let col = shape_a[1];
    let row = shape_a[0];
    if col != shape_x[0] {
        println!("vec_x shape erorr");
        std::process::exit(1);
    }
    if row != shape_y[0] {
        println!("vec_y shape erorr");
        std::process::exit(1);
    }

    for i in 0..row {
        let mut sum : f64 = 0.0;
        for j in 0..col {
            sum = sum + mat_a[[i, j]] * vec_x[[j]];
        }
        vec_y[[i]] = sum;
    }
}

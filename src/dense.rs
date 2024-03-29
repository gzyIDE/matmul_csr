use ndarray::*;
use ndarray_linalg::*;

pub fn mv_dense(mat_a: &Array2<f64>, vec_x: &Array1<f64>, vec_y: &mut Array1<f64>) {
    let shape_a = mat_a.shape();
    let shape_x = vec_x.shape();
    let shape_y = vec_y.shape();
    if shape_a[1] != shape_x[0] {
        println!("vec_x shape erorr");
        std::process::exit(1);
    }
    if shape_a[0] != shape_y[0] {
        println!("vec_y shape erorr");
        std::process::exit(1);
    }

    for i in 0..shape_a[0] {
        let mut sum : f64 = 0.0;
        for j in 0..shape_a[1] {
            sum = sum + mat_a[[i, j]] * vec_x[[j]];
        }
        vec_y[[i]] = sum;
    }
}

pub fn mm_dense(mat_a: &Array2<f64>, mat_b: &Array2<f64>, mat_z: &mut Array2<f64>) {
    let shape_a = mat_a.shape();
    let shape_b = mat_b.shape();
    let shape_z = mat_z.shape();

    if shape_a[1] != shape_b[0] {
        println!("mat_b shape error");
        std::process::exit(1);
    }
    if shape_a[0] != shape_z[0] {
        println!("mat_z shape error");
        std::process::exit(1);
    }
    if shape_b[1] != shape_z[1] {
        println!("mat_z shape error");
        std::process::exit(1);
    }

    for i in 0..shape_b[1] {
        for j in 0..shape_a[0] {
            let mut sum : f64 = 0.0;
            for k in 0..shape_a[1] {
                sum = sum + mat_a[[j, k]] * mat_b[[k,i]];
            }
            mat_z[[j,i]] = sum;
        }
    }
}

// Non-stationary systolic array (NSA) emulation
pub fn mm_systolic_nsa(mat_a: &Array2<f64>, mat_b: &Array2<f64>, mat_z: &mut Array2<f64>) {
    let shape_a = mat_a.shape();
    let shape_b = mat_b.shape();
    let shape_z = mat_z.shape();

    if shape_a[1] != shape_b[0] {
        println!("mat_b shape error");
        std::process::exit(1);
    }
    if shape_a[0] != shape_z[0] {
        println!("mat_z shape error");
        std::process::exit(1);
    }
    if shape_b[1] != shape_z[1] {
        println!("mat_z shape error");
        std::process::exit(1);
    }

    let mut ffd_x: Array2<f64> = Array::zeros((shape_a[0], shape_b[1]));
    let mut ffd_y: Array2<f64> = Array::zeros((shape_a[0], shape_b[1]));
    let mut ffq_x  : Array2<f64> = Array::zeros((shape_a[0], shape_b[1]));
    let mut ffq_y  : Array2<f64> = Array::zeros((shape_a[0], shape_b[1]));

    let cycles = (shape_a[0]-1) + shape_a[1] + (shape_b[1]-1);

    for i in 0..cycles {
        for j in 0..shape_a[0] {
            for k in 0..shape_b[1] {
                // x direction
                if k == 0 {
                    if ( i < j ) || ( shape_b[1] < i + 1 - j ) {
                        ffq_x[[j,k]] = 0.0;
                    } else {
                        let x_idx    = shape_b[1] -1 + j - i;
                        ffq_x[[j,k]] = mat_a[[j,x_idx]];
                    }
                } else {
                    ffq_x[[j,k]] = ffd_x[[j,k-1]];
                }

                // y direction
                if j == 0 {
                    if ( i < k ) || ( shape_a[0] < i + 1 - k ) {
                        ffq_y[[j,k]] = 0.0;
                    } else {
                        let y_idx    = shape_a[0] -1 + k - i;
                        ffq_y[[j,k]] = mat_b[[y_idx,k]];
                    }
                } else {
                    ffq_y[[j,k]] = ffd_y[[j-1,k]];
                }
            }
        }

        //println!("iterate: {}", i);
        //println!("ffq_x: ");
        //println!("{}", ffq_x);
        //println!("ffq_y: ");
        //println!("{}", ffq_y);
        //println!("");

        for j in 0..shape_a[0] {
            for k in 0..shape_b[1] {
                mat_z[[j,k]] = mat_z[[j,k]] + ffq_x[[j,k]] * ffq_y[[j,k]];
            }
        }

        ffd_x = ffq_x.clone();
        ffd_y = ffq_y.clone();
    }
}

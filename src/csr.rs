use ndarray::*;
use ndarray_linalg::*;
use std::fmt;

pub struct CSR {
    row     : usize,
    col     : usize,
    data    : Vec<f64>,
    row_ptr : Vec<usize>,
    col_idx : Vec<usize>
}

impl CSR {
    pub fn new() -> Self {
        CSR {
            row     : 0,
            col     : 0,
            data    : Vec::new(),
            row_ptr : Vec::new(),
            col_idx : Vec::new(),
        }
    }
}

impl fmt::Display for CSR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut csrstr = String::new();
        csrstr += &format!("row   : {}\n", self.row);
        csrstr += &format!("column: {}\n", self.col);
        csrstr += "<data>\n";
        for x in self.data.iter() {
            csrstr += &format!("{} ", x);
        }
        csrstr += "\n<row_ptr>\n";
        for x in self.row_ptr.iter() {
            csrstr += &format!("{} ", x);
        }
        csrstr += "\n<col_ptr>\n";
        for x in self.col_idx.iter() {
            csrstr += &format!("{} ", x);
        }
        write!(f, "{}", csrstr)
    }
}

pub fn d2s_csr(mat: &Array2<f64>, spmat: &mut CSR) {
    let row = mat.nrows();
    let col = mat.ncols();

    spmat.row = row;
    spmat.col = col;
    for i in 0..row {
        spmat.row_ptr.push(spmat.data.len());

        for j in 0..col {
            if mat[[i,j]] != 0.0 {
                spmat.data.push(mat[[i,j]]);
                spmat.col_idx.push(j);
            }
        }
    }
}

pub fn s2d_csr(mat: &mut Array2<f64>, spmat: &CSR) {
    let nnz = spmat.data.len();
    let row = spmat.row;

    for i in 0..row {
        let st  = spmat.row_ptr[i];
        let end = if i == row-1 { nnz }
                  else          { spmat.row_ptr[i+1] };
        
        for j in st..end {
            let col       = spmat.col_idx[j];
            mat[[i, col]] = spmat.data[j];
        }
    }
}

pub fn SpMV_csr(spmat: &CSR, vec_x: &Array1<f64>, vec_y: &mut Array1<f64>) {
    let nnz = spmat.data.len();
    let row = spmat.row;

    for i in 0..row {
        let mut sum = 0.0;
        let st      = spmat.row_ptr[i];
        let end     = if i == row-1 { nnz }
                      else          { spmat.row_ptr[i+1] };

        for j in st..end {
            let col = spmat.col_idx[j];
            sum += spmat.data[j] * vec_x[[col]];
        }

        vec_y[[i]] = sum;
    }
}

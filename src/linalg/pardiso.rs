use std::fmt::Debug;

use faer::{Index, Mat, MatRef, sparse::SparseColMatRef};
use problemo::Problem;

use crate::{
    E, I,
    linalg::solver::{LinearSolver, Solver},
};
use pardiso_wrapper::{MatrixType, PardisoInterface, Phase};

#[cfg(feature = "mkl")]
pub struct Pardiso<P: PardisoInterface> {
    col_ptrs: Vec<i32>,
    row_idx: Vec<i32>,
    values: Vec<E>,
    ps: P,
}

impl<P: PardisoInterface> Solver for Pardiso<P> {
    fn new() -> Self
    where
        Self: Sized,
    {
        let mut ps = P::new().unwrap();
        ps.pardisoinit();
        // TODO: set this more dynamically
        ps.set_matrix_type(MatrixType::RealNonsymmetric);
        ps.set_message_level(pardiso_wrapper::MessageLevel::Off);
        Self {
            col_ptrs: Vec::new(),
            row_idx: Vec::new(),
            values: Vec::new(),
            ps,
        }
    }

    fn analyze(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem> {
        (self.col_ptrs, self.row_idx, self.values) = convert_matrix_idx_type(mat);
        self.ps.set_phase(Phase::Analysis);
        self.ps.pardiso(
            self.values.as_slice(),
            self.col_ptrs.as_slice(),
            self.row_idx.as_slice(),
            &mut [],
            &mut [],
            mat.nrows() as i32,
            1,
        )?;
        Ok(())
    }

    fn factorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem> {
        self.values = mat.transpose().val().to_vec(); // Update values for refactorization
        self.ps.set_phase(Phase::NumFact);
        self.ps.pardiso(
            self.values.as_slice(),
            self.col_ptrs.as_slice(),
            self.row_idx.as_slice(),
            &mut [],
            &mut [],
            mat.nrows() as i32,
            1,
        )?;
        Ok(())
    }

    fn solve_in_place(&mut self, b: &mut faer::MatMut<crate::E>) -> Result<(), Problem> {
        self.ps.set_phase(Phase::SolveIterativeRefine);
        let mut b_vec: Vec<E> = (0..b.ncols())
            .flat_map(|j| b.as_ref().col(j).iter().copied())
            .collect();
        let mut x_vec = vec![0.0 as E; b.nrows() * b.ncols()];

        self.ps.pardiso(
            self.values.as_slice(),
            self.col_ptrs.as_slice(),
            self.row_idx.as_slice(),
            b_vec.as_mut_slice(),
            x_vec.as_mut_slice(),
            b.nrows() as i32,
            b.ncols() as i32,
        )?;

        let x = MatRef::from_column_major_slice(&x_vec, b.nrows(), b.ncols());
        b.copy_from(&x);
        Ok(())
    }
}

impl<P: PardisoInterface> LinearSolver for Pardiso<P> {}

/// Converts a CSC matrix (faer, 0-based) to CSR format (PARDISO, 1-based).
fn convert_matrix_idx_type<T>(mat: SparseColMatRef<I, E>) -> (Vec<T>, Vec<T>, Vec<E>)
where
    T: TryFrom<usize> + Debug,
    T::Error: Debug,
    I: Index,
{
    let n = mat.nrows();
    let nnz = mat.compute_nnz();

    // Count entries per row to build CSR row pointers
    let mut row_counts = vec![0usize; n];
    for &r in mat.row_idx() {
        row_counts[r] += 1;
    }

    // Build 1-based CSR row pointers (ia)
    let mut ia = vec![0usize; n + 1];
    for i in 0..n {
        ia[i + 1] = ia[i] + row_counts[i];
    }
    let row_ptrs: Vec<T> = ia.iter().map(|&x| T::try_from(x + 1).unwrap()).collect();

    // Fill CSR column indices (ja) and values, sorted by column within each row
    let mut ja = vec![0usize; nnz];
    let mut vals = vec![0.0f64; nnz];
    let mut row_pos = ia[..n].to_vec(); // current fill position per row

    for col in 0..mat.ncols() {
        let col_start = mat.col_ptr()[col];
        let col_end = mat.col_ptr()[col + 1];
        for idx in col_start..col_end {
            let row = mat.row_idx()[idx];
            let pos = row_pos[row];
            ja[pos] = col + 1; // 1-based column index
            vals[pos] = mat.val()[idx];
            row_pos[row] += 1;
        }
    }

    let col_idx: Vec<T> = ja.iter().map(|&x| T::try_from(x).unwrap()).collect();

    (row_ptrs, col_idx, vals)
}

#[cfg(feature = "mkl")]
pub type MKLPardiso = Pardiso<pardiso_wrapper::MKLPardisoSolver>;
#[cfg(feature = "panua")]
pub type PanuaPardiso = Pardiso<pardiso_wrapper::PanuaPardisoSolver>;

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[cfg(feature = "mkl")]
    #[rstest]
    fn test_mkl(#[values("Trefethen 20b")] mat_name: &str) {
        let mat = loaders::mtx::get_matrix_by_name::<I, E>(mat_name, true);

        let mut solver = MKLPardiso::new();
        solver.analyze(mat.as_ref()).unwrap();
        solver.factorize(mat.as_ref()).unwrap();

        let n = mat.ncols();
        let mut b = Mat::zeros(n, 1);
        for i in 0..n {
            b[(i, 0)] = E::from(i as f64 + 1.0); // Example right-hand side
        }

        let x = solver.solve(b.as_ref()).unwrap();

        // Here you would typically compare `b` to a known solution or check residuals
        let err = mat * &x - &b;
        assert!(err.norm_l2() < 1e-10);
    }

    #[cfg(feature = "panua")]
    #[rstest]
    fn test_panua(#[values("Trefethen 20b")] mat_name: &str) {
        let mat = loaders::mtx::get_matrix_by_name::<I, E>(mat_name, true);

        let mut solver = PanuaPardiso::new();
        solver.analyze(mat.as_ref()).unwrap();
        solver.factorize(mat.as_ref()).unwrap();

        let n = mat.ncols();
        let mut b = Mat::zeros(n, 1);
        for i in 0..n {
            b[(i, 0)] = E::from(i as f64 + 1.0); // Example right-hand side
        }

        solver.solve_in_place(&mut b.as_mut()).unwrap();

        // Here you would typically compare `b` to a known solution or check residuals
        let err = mat * &x - &b;
        assert!(err.norm_l2() < 1e-10);
    }
}

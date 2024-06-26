#![allow(dead_code)]
use crate::algebra::*;
use crate::solver::core::kktsolvers::direct::DirectLDLSolver;
use crate::solver::core::CoreSettings;
use libloading::{Library, library_filename, Symbol};

// What is the big difference between using symbol and RawSymbol?
#[cfg(target_family="unix")]
use libloading::os::unix::Symbol as RawSymbol;
#[cfg(target_family="windows")]
use libloading::os::windows::Symbol as RawSymbol;
use std::ffi::{c_int,c_void};
use std::marker::PhantomData; // Maybe one does not need this?

// Defining Functions
type Init = unsafe extern "C" fn(
                            pt: *mut c_void,
                            mtype: *mut c_int,
                            solver: *mut c_int,
                            iparm: *mut c_int,
                            dparm: *mut f64,
                            error: *mut c_int);
type Pardiso = unsafe extern "C" fn(
                            pt: *mut c_void,
                            maxfct: *mut c_int,
                            mnum: *mut c_int,
                            mtype: *mut c_int,
                            phase: *mut c_int,
                            n: *mut c_int,
                            a: *const c_void,
                            ia: *const c_int,
                            ij: *const c_int,
                            perm: *mut c_int,
                            nrhs: *mut c_int,
                            iparm: *mut c_int,
                            msglvl: *mut c_int,
                            b: *const c_void,
                            x: *mut c_void,
                            error: *mut c_int,
                            dparm: *mut f64);

struct PardisoSolver {
    library: Library, // We need to this so taht the Symbols stay in scope?
    pardisoinit: RawSymbol<Init>,
    pardiso: RawSymbol<Pardiso>,
}

impl PardisoSolver{
    unsafe fn new() -> PardisoSolver {
        
        // Load library at runtime. 
        let library = unsafe {Library::new(library_filename("pardiso")).unwrap()};
        // Load pardisoinit function
        let pardisoinit: Symbol<Init> = unsafe{library.get(b"pardisoinit").unwrap()};
        let pardisoinit = pardisoinit.into_raw();
        // Load pardiso function
        let pardiso: Symbol<Pardiso> = unsafe { library.get(b"pardiso").unwrap()};
        let pardiso = pardiso.into_raw();
        
        PardisoSolver {
            library,
            pardisoinit,
            pardiso,
        }
    }
}

pub struct PanuaPardisoDirectLDLSolver<T: FloatT> {
    // 1-indexed indices as i32.
    ia: Vec<i32>, // 1-indexed rowptr   (CSR format)
    ja: Vec<i32>, // 1-indexed columval (CSR format)

    pardiso_solver: PardisoSolver,

    pt: Vec<isize>, // Point to handle

    iparm: Vec<i32>,
    dparm: Vec<f64>,

    mtype: i32, // Should always be -2 (Real and Symmetric indefinite)

    msglvl: i32, // Standardly 0. However, set to 1 for debugging
    
    maxfct: i32, // Number of saved factorizations. Standard is to set to 1.
    mnum: i32,   // Which factorization to use. Standard is to set to 1.

    n: i32,     // Number of equations. Should be equal to number of rows in KKT

    perm: Vec<i32>, // Possibly use specified permutation. For now we do not allow this.

    error: i32, 

    phantom: PhantomData<T>, // So that clippy does not complain of unsued "T"
}

impl<T> PanuaPardisoDirectLDLSolver<T>
where
    T: FloatT,
{
    pub fn new(KKT: &CscMatrix<T>, _Dsigns: &[i8], _settings: &CoreSettings<T>) -> Self {
        let dim = KKT.nrows();

        assert!(dim == KKT.ncols(), "KKT matrix is not square");

        let mut pt: Vec<isize> = vec![0; 64];    // Point to handle
        let mut iparm: Vec<i32> = vec![0; 64];   // Integer parameters
        let mut dparm: Vec<f64> = vec![0.0; 64]; // Double parameters
        let mut mtype: i32 = -2; // Real and Symmetric indefinite (Real Sym. Quasi-definite)
        let mut phase: i32 = 11; // Symbolic phase
        let mut msglvl: i32 = 0; // No message
        let mut maxfct: i32 = 1; // Only one factorization at a time
        let mut mnum: i32 = 1;   // We use the first (and only) factorization
        let mut n: i32 = KKT.m as i32; // no. equations i.e. no. rows in KKT
        let mut error: i32 = 0;
        let mut solver = 0; // Sparse Direct Solver

        // Create PardisoSolver
        let pardiso_solver = unsafe {PardisoSolver::new()};
        
        // Initialize pt handle, iparm, and dparm
        unsafe {(pardiso_solver.pardisoinit)(
                pt.as_mut_ptr() as *mut _ as *mut c_void,
                &mut mtype,
                &mut solver,
                iparm.as_mut_ptr(),
                dparm.as_mut_ptr(),
                &mut error,
            )
        };
        
        iparm[8 - 1] = -99; // No IR - Has to be set after init!
        // For now we do not allow user specified permutation
        let mut perm: Vec<i32> = vec![0; 1];    
        // Create new one-indexed row and column pointers
        let mut ia: Vec<i32> = vec![0; KKT.colptr.len()];
        let mut ja: Vec<i32> = vec![0; KKT.rowval.len()];
        for i in 0..ia.len() {
            ia[i] = 1 + KKT.colptr[i] as i32;
        }
        for j in 0..ja.len() {
            ja[j] = 1 + KKT.rowval[j] as i32;
        }
        
        // Perform symbolic factorization
        // RHS and X are not used in the symbolic factorization so we create dummies
        let mut dummy: Vec<f64> = vec![0.0; 1];  
        let mut nhrs = 1;
        unsafe {(pardiso_solver.pardiso)(
                pt.as_mut_ptr() as *mut _ as *mut c_void,
                &mut maxfct,
                &mut mnum,
                &mut mtype,
                &mut phase,
                &mut n,
                KKT.nzval.as_ptr()  as *const _ as *const c_void,
                ia.as_ptr() as *const _ as *const i32, // Need to change to 1-indexing?
                ja.as_ptr() as *const _ as *const i32, // Need to change to 1-indexing?
                perm.as_mut_ptr(),
                &mut nhrs,
                iparm.as_mut_ptr(),
                &mut msglvl,
                dummy.as_ptr() as *const _ as *const c_void,
                dummy.as_mut_ptr() as *mut _ as *mut c_void,
                &mut error,
                dparm.as_mut_ptr(),
            )
        };
        
        Self {
            ia,
            ja,
            pardiso_solver, 
            pt,
            iparm,
            dparm,
            mtype,
            msglvl,
            maxfct,
            mnum,
            n,
            perm,
            error,
            phantom: PhantomData,
            }
    }
}

impl<T> DirectLDLSolver<T> for PanuaPardisoDirectLDLSolver<T>
where
    T: FloatT,
{
    fn update_values(&mut self, _index: &[usize], _values: &[T]) {
        // No-op
    }

    fn scale_values(&mut self, _index: &[usize], _scale: T) {
        // No-op
    }

    fn offset_values(&mut self, _index: &[usize], _offset: T, _signs: &[i8]) {
        // No-op
    }

    fn solve(&mut self, _kkt: &CscMatrix<T>, x: &mut [T], b: &[T]) {
        
        let mut phase = 33; // Solve phase (No IR as iparm[8-1]=-99)
        let mut nrhs = 1;   // Only one RHS

        unsafe {
            (self.pardiso_solver.pardiso)(
                self.pt.as_mut_ptr() as *mut _ as *mut c_void,
                &mut self.maxfct,
                &mut self.mnum,
                &mut self.mtype,
                &mut phase,
                &mut self.n,
                _kkt.nzval.as_ptr()  as *const _ as *const c_void,
                self.ia.as_ptr() as *const _ as *const i32, // Need to change to 1-indexing?
                self.ja.as_ptr() as *const _ as *const i32, // Need to change to 1-indexing?
                self.perm.as_mut_ptr(),
                &mut nrhs,
                self.iparm.as_mut_ptr(),
                &mut self.msglvl,
                b.as_ptr() as *const _ as *const c_void,
                x.as_mut_ptr() as *mut _ as *mut c_void,
                &mut self.error,
                self.dparm.as_mut_ptr(),
            )
        }
    }

    fn refactor(&mut self, _kkt: &CscMatrix<T>) -> bool {
        
        let mut phase = 22; // Perform Numerical Factorization
        let mut nrhs = 1;   // Number of RHS. Does not matter in a numerical factorization
        let mut dummy: Vec<f64> = vec![0.0; 1]; // Not used in the numerical factorization
        unsafe {
            (self.pardiso_solver.pardiso)(
                self.pt.as_mut_ptr() as *mut _ as *mut c_void,
                &mut self.maxfct,
                &mut self.mnum,
                &mut self.mtype,
                &mut phase,
                &mut self.n,
                _kkt.nzval.as_ptr()  as *const _ as *const c_void,
                self.ia.as_ptr() as *const _ as *const i32, 
                self.ja.as_ptr() as *const _ as *const i32,
                self.perm.as_mut_ptr(),
                &mut nrhs,
                self.iparm.as_mut_ptr(),
                &mut self.msglvl,
                dummy.as_ptr() as *const _ as *const c_void,
                dummy.as_mut_ptr() as *mut _ as *mut c_void,
                &mut self.error,
                self.dparm.as_mut_ptr(),
            )
        }
        //    0No error.
        //   -1 Input inconsistent.
        //   -2 Not enough memory.
        //   -3 Reordering problem.
        //   -4 Zero pivot, numerical fact. or iterative refinement problem.
        //   -5 Unclassified (internal) error.
        //   -6 Preordering failed (matrix types 11, 13 only).
        //   -7 Diagonal matrix problem.
        //   -8 32-bit integer overflow problem.
        //  -10 No license file pardiso.lic found.
        //  -11 License is expired.
        //  -12 Wrong username or hostname.
        // -100 Reached maximum number of Krylov-subspace iteration in iterative solver. 
        // -101 No sufficient convergence in Krylov-subspace iteration within 25 iterations.
        // -102 Error in Krylov-subspace iteration.
        // -103 Break-Down in Krylov-subspace iteration.
        if self.error == 0 {
            return true;
        } else {
            return false;
        }

    }

    fn required_matrix_shape() -> MatrixTriangle {
        // Panua needs triu of a CSR matrix and KKT is CSC. 
        // Making KKT Tril manes that we can get triu CSR by swapping rows/cols of tril-KKT.
        MatrixTriangle::Tril
    }
}

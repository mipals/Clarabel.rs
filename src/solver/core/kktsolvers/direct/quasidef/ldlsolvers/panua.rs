#![allow(non_snake_case)]
#![allow(dead_code)]
use crate::algebra::*;
use crate::solver::core::kktsolvers::direct::DirectLDLSolver;
use crate::solver::core::CoreSettings;

use libloading::{Library, library_filename, Symbol};
use libloading::os::unix::Symbol as RawSymbol;
use std::ffi::{c_int,c_void};
use std::marker::PhantomData;

// Defining loading 
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
    library: Library,
    pardisoinit: RawSymbol<Init>,
    pardiso: RawSymbol<Pardiso>,
}

impl PardisoSolver{
    unsafe fn new() -> PardisoSolver {
        
        // Load library at runtime
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
    ia: Vec<i32>, // 1-indexed rowptr   (CSR format)
    ja: Vec<i32>, // 1-indexed columval (CSR format)

    pardiso_solver: PardisoSolver,

    pt: Vec<isize>, // Point to handle

    iparm: Vec<i32>,
    dparm: Vec<f64>,

    mtype: i32,

    msglvl: i32,
    
    maxfct: i32,
    mnum: i32,

    n: i32,

    perm: Vec<i32>,

    error: i32,

    phantom: PhantomData<T>,
}

impl<T> PanuaPardisoDirectLDLSolver<T>
where
    T: FloatT,
{
    pub fn new(KKT: &CscMatrix<T>, _Dsigns: &[i8], _settings: &CoreSettings<T>) -> Self {
        let dim = KKT.nrows();

        assert!(dim == KKT.ncols(), "KKT matrix is not square");

        // occasionally we find that the default AMD parameters give a bad ordering, particularly
        // for some big matrices.  In particular, KKT conditions for QPs are sometimes worse
        // than their SOC counterparts for very large problems.   This is because the SOC form
        // is artificially "big", with extra rows, so the dense row threshold is effectively a
        // different value.   We fix a bit more generous AMD_DENSE here, which should perhaps
        // be user-settable.

        //make a logical factorization to fix memory allocations

        let mut pt: Vec<isize> = vec![0; 64];
        /* Pardiso control parameters. */
        let mut iparm: Vec<i32> = vec![0; 64];
        let mut dparm: Vec<f64> = vec![0.0; 64];

        let mut mtype: i32 = -2; // Real and Symmetric indefinite (Real Sym. Quasi-definite)
        
        let mut phase: i32 = 11; // Analysis/Symbolic phase
        let mut msglvl: i32 = 0; // No message
        
        let mut maxfct: i32 = 1; // Only one factorization at a time
        let mut mnum: i32 = 1;  // We use the only factorization
        
        let mut n: i32 = KKT.m as i32; // no. equations i.e. no. rows in KKT
        
        let mut error: i32 = 0;
        // Initialize Pardiso
        let pardiso_solver = unsafe {PardisoSolver::new()};
        
        let mut solver = 0; // Sparse Direct Solver
        unsafe {(pardiso_solver.pardisoinit)(
                pt.as_mut_ptr() as *mut _ as *mut c_void,
                &mut mtype,
                &mut solver,
                iparm.as_mut_ptr(),
                dparm.as_mut_ptr(),
                &mut error,
            )
        };
        
        iparm[8 - 1] = -99; // No IR - Has to be set after init
        // Perform symbolic 
        // let mut perm: Vec<i32> = vec![0; n];
        // For now we do not allow user-given permutation
        let mut perm: Vec<i32> = vec![0; 1];    
        // The entries have to be one-indexed.
        let mut ia: Vec<i32> = vec![0; KKT.colptr.len()];
        let mut ja: Vec<i32> = vec![0; KKT.rowval.len()];


        for i in 0..ia.len() {
            ia[i] = 1 + KKT.colptr[i] as i32;
        }
        for j in 0..ja.len() {
            ja[j] = 1 + KKT.rowval[j] as i32;
        }
        // println!("len={}",ia[0]); // Have to be 1
        // println!("len={}",ja[0]); // Have to be 1


        let mut dummy: Vec<f64> = vec![0.0; 1]; // Not used in the symbolic factorization
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

        println!("iparm11 = {}", iparm[11]); // This should be 8??

        println!("Number of nonzeros in factors={}", iparm[17]);

        println!("{error}");

        
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
        //Update values that are stored within
        //the reordered copy held internally by QDLDL.
        // No-op
    }

    fn scale_values(&mut self, _index: &[usize], _scale: T) {
        // No-op
    }

    fn offset_values(&mut self, _index: &[usize], _offset: T, _signs: &[i8]) {
        // self.factors.offset_values(index, offset, signs);
        // No-op
    }

    fn solve(&mut self, _kkt: &CscMatrix<T>, x: &mut [T], b: &[T]) {
        // NB: QDLDL solves in place
        // x.copy_from(b);
        // self.factors.solve(x);
        let mut phase = 33; // Solve and IR (but IR is disabled!)
        let mut nrhs = 1;   // For now only allow a single rhs

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
        //QDLDL has maintained its own version of the permuted
        //KKT matrix through custom update/scale/offset methods,
        //so we ignore the KKT matrix provided by the caller
        let mut phase = 22; // Numerical Factorization
        let mut nrhs = 1;   // dummy
        let mut dummy: Vec<f64> = vec![0.0; 1]; // Not used in the symbolic factorization
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
                dummy.as_ptr() as *const _ as *const c_void,
                dummy.as_mut_ptr() as *mut _ as *mut c_void,
                &mut self.error,
                self.dparm.as_mut_ptr(),
            )
        }
        // if self.error == 0 {
        //     return true;
        // } else {
        //     return false;
        // }
        true
    }

    fn required_matrix_shape() -> MatrixTriangle {
        // Tril since we want triu of a CSR matrix and KKT is CSC
        MatrixTriangle::Tril
    }
}




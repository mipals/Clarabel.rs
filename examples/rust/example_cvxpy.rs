#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;


fn main() {
    let h = vec![0., 0., 12., 6., 0., 0., 0., 2.];
    let c = vec![0., 0., 0.];
    let P = CscMatrix::zeros((3, 3));
     //direct from sparse data
    let G = CscMatrix::new(
        8,                               // m
        3,                               // n
        vec![0, 3, 7, 10],          // rowval
        vec![0, 2, 4, 1, 3, 5, 6, 4, 5, 6],                   // colptr
        vec![-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 72.00195312, -1.0, 1.0], // nzval
    );
    
    let cones = [NonnegativeConeT(5), SecondOrderConeT(3)];

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &c, &G, &h, &cones, settings);

    solver.solve();
}

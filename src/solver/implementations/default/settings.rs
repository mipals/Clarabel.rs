use crate::algebra::*;
use crate::solver::core::traits::Settings;
use derive_builder::Builder;

#[cfg(feature = "julia")]
use serde::{Deserialize, Serialize};

/// Standard-form solver type implementing the [Settings](crate::solver::core::traits::Settings) trait

#[derive(Builder, Debug, Clone)]
#[cfg_attr(feature = "julia", derive(Serialize, Deserialize))]
pub struct DefaultSettings<T: FloatT> {
    #[builder(default = "50")]
    pub max_iter: u32,

    #[builder(default = "f64::INFINITY")]
    pub time_limit: f64,

    #[builder(default = "true")]
    pub verbose: bool,

    #[builder(default = "(1e-8).as_T()")]
    pub tol_gap_abs: T,

    #[builder(default = "(1e-8).as_T()")]
    pub tol_gap_rel: T,

    #[builder(default = "(1e-5).as_T()")]
    pub tol_feas: T,

    #[builder(default = "(1e-8).as_T()")]
    pub tol_infeas_abs: T,

    #[builder(default = "(1e-8).as_T()")]
    pub tol_infeas_rel: T,

    #[builder(default = "(0.99).as_T()")]
    pub max_step_fraction: T,

    // data equilibration
    #[builder(default = "true")]
    pub equilibrate_enable: bool,

    #[builder(default = "10")]
    pub equilibrate_max_iter: u32,

    #[builder(default = "(1e-4).as_T()")]
    pub equilibrate_min_scaling: T,

    #[builder(default = "(1e+4).as_T()")]
    pub equilibrate_max_scaling: T,

    // only support direct / qdldl at the moment
    #[builder(default = "true")]
    pub direct_kkt_solver: bool,
    #[builder(default = r#""qdldl".to_string()"#)]
    pub direct_solve_method: String,

    // static regularization parameters
    #[builder(default = "true")]
    pub static_regularization_enable: bool,
    #[builder(default = "(1e-8).as_T()")]
    pub static_regularization_eps: T,

    // dynamic regularization parameters
    #[builder(default = "true")]
    pub dynamic_regularization_enable: bool,

    #[builder(default = "(1e-13).as_T()")]
    pub dynamic_regularization_eps: T,

    #[builder(default = "(2e-7).as_T()")]
    pub dynamic_regularization_delta: T,

    // iterative refinement (for direct solves)
    #[builder(default = "true")]
    pub iterative_refinement_enable: bool,

    #[builder(default = "(1e-10).as_T()")]
    pub iterative_refinement_reltol: T,

    #[builder(default = "(1e-10).as_T()")]
    pub iterative_refinement_abstol: T,

    #[builder(default = "10")]
    pub iterative_refinement_max_iter: u32,

    #[builder(default = "(2.0).as_T()")]
    pub iterative_refinement_stop_ratio: T,
}

impl<T> Default for DefaultSettings<T>
where
    T: FloatT,
{
    fn default() -> DefaultSettings<T> {
        DefaultSettingsBuilder::<T>::default().build().unwrap()
    }
}

impl<T> Settings<T> for DefaultSettings<T>
where
    T: FloatT,
{
    //NB: CoreSettings is typedef'd to DefaultSettings
    fn core(&self) -> &DefaultSettings<T> {
        self
    }
    fn core_mut(&mut self) -> &mut DefaultSettings<T> {
        self
    }
}

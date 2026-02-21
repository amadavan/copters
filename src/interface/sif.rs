use faer::{Col, sparse::{SparseColMat, Triplet}};
use sif_rs::SIF;
use problemo::{Problem, common::IntoCommonProblem};

use crate::{E, I, lp::LinearProgram, qp::QuadraticProgram};

pub trait TryFromSIF {
    type Output;
    fn try_from_sif(sif: &SIF) -> Result<Self::Output, Problem>;
}

impl TryFromSIF for LinearProgram {
    type Output = Self;

    fn try_from_sif(sif: &SIF) -> Result<Self::Output, Problem> {
        let data = parse_sif(sif)?;
        Ok(Self::new(
            data.c,
            data.A,
            data.b,
            data.l,
            data.u,
        ))
    }
}

impl TryFromSIF for QuadraticProgram {
    type Output = Self;

    fn try_from_sif(sif: &SIF) -> Result<Self::Output, Problem> {
        let data = parse_sif(sif)?;
        
        #[allow(non_snake_case)]
        let Q = data.Q.ok_or_else(|| "Quadratic term is required for a quadratic program".gloss())?;
        Ok(Self::new(
            Q,
            data.c,
            data.A,
            data.b,
            data.l,
            data.u,
        ))
    }
}

#[allow(non_snake_case)]
struct SifData {
    c: Col<E>,
    A: SparseColMat<I, E>,
    b: Col<E>,
    l: Col<E>,
    u: Col<E>,
    Q: Option<SparseColMat<I, E>>,
}

fn parse_sif(sif: &SIF) -> Result<SifData, Problem> {
    // Map variable and constraint names to their respective internal indices
    // Use BTreeSet/BTreeMap for deterministic ordering of indices
    let map_var_idx: std::collections::BTreeMap<_, _> = sif.get_cols().into_iter().map(|(var_name, _)| var_name.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().enumerate().map(|(i, var_name)| (var_name, i)).collect();
    let map_con_idx: std::collections::BTreeMap<_, _> = sif.get_rows().into_iter().filter(|(_, rhs_type)| rhs_type != &&sif_rs::types::RowType::N).map(|(con_name, _)| con_name.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().enumerate().map(|(i, con_name)| (con_name, i)).collect();

    let (n_var, n_con) = (map_var_idx.len(), map_con_idx.len());

    // Get number of slack variables
    let n_slack = sif.get_rows().iter().filter(|(_, rhs_type)| **rhs_type == sif_rs::types::RowType::L || **rhs_type == sif_rs::types::RowType::G).count();

    // Construct the objective function
    let mut c = Col::zeros(n_var + n_slack);
    sif.get_entries().iter().filter(|((con, _var), _)| 
            // Filter out non-objective function coefficients
            sif.get_rows().get(con) == Some(&&sif_rs::types::RowType::N))
        .for_each(|((_con, var), &val)| {
            let j = map_var_idx[var];
            c[j] = E::from(val);
        });

    // Construct the right-hand side vector
    let b = sif.get_rhs().into_iter().filter(|(con, _val)| sif.get_rows().get(*con) != Some(&&sif_rs::types::RowType::N)).map(|(con, val)| {
        let i = map_con_idx[con];
        (i, val)
    }).fold(Col::zeros(n_con), |mut b, (i, val)| {
        b[i] = E::from(*val);
        b
    });

    let a_triplets = sif.get_entries().iter().filter(|((con, _var), val)| {
            // Filter out zero coefficients and objective function coefficients
            if **val == 0. {
                return false;
            }

            if sif.get_rows().get(con) == Some(&&sif_rs::types::RowType::N) {
                return false;
            }

            true
        }).map(|(i, &val)| {
            let (i, j) = (map_con_idx[&i.0], map_var_idx[&i.1]);
            Triplet::new(I::from(i), I::from(j), E::from(val))
        }).collect::<Vec<_>>();

    // Construct bounds
    let mut l = Col::<E>::zeros(n_var + n_slack);
    let mut u = E::INFINITY * Col::<E>::ones(n_var + n_slack);
    sif.get_bounds().into_iter().for_each(|(var_name, (bound_type, val))| {
            let j = map_var_idx[var_name];

            match bound_type {
                sif_rs::types::BoundType::Lo => {
                    l[j] = E::from(*val);
                }
                sif_rs::types::BoundType::Up => {
                    u[j] = E::from(*val);
                }
                sif_rs::types::BoundType::Fr => {
                    l[j] = -E::INFINITY;
                    u[j] = E::INFINITY;
                }
                sif_rs::types::BoundType::Mi => {
                    l[j] = -E::INFINITY;
                    u[j] = E::from(0.);
                }
                sif_rs::types::BoundType::Pl => {
                    l[j] = E::from(0.);
                    u[j] = E::INFINITY;
                }
                sif_rs::types::BoundType::Fx => {
                    // TODO: cannot currently handle fixed variables properly because we need to ensure the initial iterate is strictly feasible. For now, we just add a small tolerance around the fixed value.
                    l[j] = E::from(*val - 0.01);
                    u[j] = E::from(*val + 0.01);
                }
                // sif_rs::types::BoundType::Bv => {
                //     l[j] = E::from(0.);
                //     u[j] = E::from(1.);
                // }
                // sif_rs::types::BoundType::Li => {
                //     l[j] = E::from(*val);
                //     u[j] = E::INFINITY;
                // }
                // sif_rs::types::BoundType::Ui => {
                //     l[j] = -E::INFINITY;
                //     u[j] = E::from(*val);
                // }
                // sif_rs::types::BoundType::Sc => {
                //     // Special case for semi-continuous variables: either 0 or above a threshold
                //     l[j] = E::from(*val);
                //     u[j] = E::INFINITY;
                // }
                _ => panic!("Unsupported bound type: {:?}", bound_type),
            }
        });

    // Add slack variable coefficients to the constraint matrix
    let slack_triplets = map_con_idx.iter()
        .map(| (con_name, &i)| (sif.get_rows()[con_name], i))
        .filter(|(con_type, _)| *con_type == sif_rs::types::RowType::L || *con_type == sif_rs::types::RowType::G)
        .enumerate()
        .map(|(i, (con_type, j))| {
            match con_type {
                sif_rs::types::RowType::L => Triplet::new(I::from(j), I::from(n_var + i), E::from(1.)),
                sif_rs::types::RowType::G => Triplet::new(I::from(j), I::from(n_var + i), E::from(-1.)),
                _ => unreachable!(),
            }
        });

    let a_triplets = a_triplets.into_iter().chain(slack_triplets).collect::<Vec<_>>();

    #[allow(non_snake_case)]
    let A = SparseColMat::try_new_from_triplets(n_con, n_var + n_slack, a_triplets.as_slice()).unwrap();

    #[allow(non_snake_case)]
    let Q = {
        let q_triplet = sif.get_quadratic().into_iter().map(|((var1, var2), coeff)| {
            let j1 = map_var_idx[var1];
            let j2 = map_var_idx[var2];
            Triplet::new(I::from(j1), I::from(j2), E::from(*coeff))
        }).collect::<Vec<_>>();
        SparseColMat::try_new_from_triplets(n_var, n_var, &q_triplet)
    }.unwrap();

    Ok(SifData {
        c,
        A,
        b,
        l,
        u,
        Q: if Q.compute_nnz() > 0 { Some(Q) } else { None },
    })
}
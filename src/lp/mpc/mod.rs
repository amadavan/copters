use crate::{IterativeSolver, Status};

pub struct MehrotraPredictorCorrector {
}

impl MehrotraPredictorCorrector {
    const DEFAULT_MAX_ITER: usize = 100;

    pub fn new() -> Self {
        Self { }
    }
}

impl IterativeSolver for MehrotraPredictorCorrector {
    fn get_max_iter() -> usize {
        Self::DEFAULT_MAX_ITER
    }

    fn initialize(&mut self) {
        // Initialization code here
    }

    fn iterate(&mut self) {
        // Iteration step code here
    }

    fn get_status(&self) -> Status {
        // Convergence check code here
        Status::InProgress
    }
}
#![feature(fn_traits)]

pub mod mps;
pub mod mtx;
pub mod netlib;
pub mod sif;

fn get_data_dir() -> String {
    format!("{}/data", env!("CARGO_MANIFEST_DIR"))
}

fn get_cache_dir() -> String {
    format!("{}/artifacts", env!("CARGO_MANIFEST_DIR"))
}

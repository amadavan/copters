#[cfg(test)]
#[macro_use]
extern crate macros;

pub mod mtx;
pub mod netlib;

fn get_data_dir() -> String {
    format!("{}/data", env!("CARGO_MANIFEST_DIR"))
}

fn get_cache_dir() -> String {
    format!("{}/artifacts", env!("CARGO_MANIFEST_DIR"))
}

use faer::Index;
use faer::sparse::{SparseColMat, Triplet};
use faer::traits::ComplexField;
use flate2::bufread::GzDecoder;
use matrix_market_rs::MtxData;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write, copy};
use std::path::Path;
use std::sync::LazyLock;
use tempfile::NamedTempFile;

pub static MATRICES_URL_MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        (
            "Trefethen 20b",
            "https://suitesparse-collection-website.herokuapp.com/MM/JGD_Trefethen/Trefethen_20b.tar.gz",
        ),
        (
            "nd3k",
            "http://sparse-files.engr.tamu.edu/MM/ND/nd3k.tar.gz",
        ),
        (
            "bundle1",
            "https://suitesparse-collection-website.herokuapp.com/MM/Lourakis/bundle1.tar.gz",
        ),
        (
            "smt",
            "https://suitesparse-collection-website.herokuapp.com/MM/TKK/smt.tar.gz",
        ),
        (
            "BenElechi1",
            "https://suitesparse-collection-website.herokuapp.com/MM/BenElechi/BenElechi1.tar.gz",
        ),
        (
            "consph",
            "https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz",
        ),
        (
            "bundle_adj",
            "https://suitesparse-collection-website.herokuapp.com/MM/Mazaheri/bundle_adj.tar.gz",
        ),
        (
            "nd24k",
            "http://sparse-files.engr.tamu.edu/MM/ND/nd24k.tar.gz",
        ),
        (
            "af shell7",
            "http://sparse-files.engr.tamu.edu/MM/Schenk_AFE/af_shell7.tar.gz",
        ),
        (
            "G3 circuit",
            "http://sparse-files.engr.tamu.edu/MM/AMD/G3_circuit.tar.gz",
        ),
    ])
});

pub fn get_matrix_by_name<I: Index + std::convert::From<usize>, E: ComplexField>(
    name: &str,
    sym: bool,
) -> SparseColMat<I, E> {
    let cache_dir = format!("{}/artifacts", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

    let file_name = format!("{}/{}.tar.gz", cache_dir, name);
    if !Path::new(&file_name).exists() {
        println!("Downloading file {}", name);
        let url = MATRICES_URL_MAP.get(name).expect("Unknown matrix");
        let response = reqwest::blocking::get(*url).expect("Failed to download matrix");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_name)
            .expect("Failed to create file");
        copy(&mut response.take(usize::MAX as u64), &mut file).expect("Unable to copy file.");
        file.sync_all().expect("Failed to sync file");
    }

    let file = File::open(&file_name).expect("Failed to read file");
    let buf_reader = BufReader::new(file);
    let gz = GzDecoder::new(buf_reader);
    let mut archive = tar::Archive::new(gz);
    let mut entries = archive.entries().unwrap();

    let mut entry = entries.next().unwrap().unwrap();

    let mut mtx_bytes = Vec::new();
    entry
        .read_to_end(&mut mtx_bytes)
        .expect("Failed to read matrix file");

    // Create a named temporary file
    let mut tmpfile = NamedTempFile::new().expect("Failed to create temp file");
    tmpfile
        .write_all(&mtx_bytes)
        .expect("Failed to write matrix data");

    // Use the file path with your loader
    let mtx =
        MtxData::<f64, 2>::from_file(tmpfile.path()).expect("Failed to parse Matrix Market data");

    // Now you can destructure mtx as needed, for example:
    let MtxData::Sparse([nrows, ncols], coord, val, _) = mtx else {
        panic!("Matrix is not in sparse format");
    };

    SparseColMat::try_new_from_triplets(
        nrows,
        ncols,
        &if sym {
            coord
                .iter()
                .zip(&val)
                .flat_map(|(&[row, col], &val)| {
                    let val = if row == col { val / 2.0 } else { val };
                    [
                        Triplet::new(row.into(), col.into(), E::from_f64_impl(val)),
                        Triplet::new(col.into(), row.into(), E::from_f64_impl(val)),
                    ]
                })
                .collect::<Vec<_>>()
        } else {
            coord
                .iter()
                .zip(&val)
                .map(|(&[row, col], &val)| {
                    Triplet::new(row.into(), col.into(), E::from_f64_impl(val))
                })
                .collect::<Vec<_>>()
        },
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::prelude::*;

    #[value_parameterized_test(values = ["Trefethen 20b"])] //, "bundle1", "nd3k"])]
    fn test_matrix_symmetry(name: &'static str) {
        let mat = get_matrix_by_name::<usize, f64>(name, true);
        let mat_dense = mat.to_dense();
        let error = (&mat_dense - &mat_dense.transpose()).norm_l2();
        println!("Error = {:e}", error);
        assert!(error < 1e-12, "Matrix not symmetric");
    }
}

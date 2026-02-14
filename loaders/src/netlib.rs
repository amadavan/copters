/// Provides utilities for downloading, caching, and decompressing Netlib LP test cases in EMPS
/// format, and converting them to MPS format for use with the `mps` crate.
///
/// # FFI
/// This module uses FFI to interface with C functions for decompressing `.emps` files:
/// - `emps_init`: Initializes the EMPS decompression environment.
/// - `process_from_filename`: Processes a compressed EMPS file and outputs the decompressed MPS to
///   stdout.
///
/// # Constants
/// - `URL`: Base URL for Netlib LP data.
///
/// # Structs
/// - `NetlibLoader`: Handles downloading, caching, and decompressing Netlib LP test cases.
///     - `new()`: Creates a new loader, initializing the cache directory and FFI arguments.
///     - `download_compressed(&self, name: &str)`: Downloads the compressed EMPS file if not
///       already cached.
///     - `decompress_mps(&self, emps_path: &str)`: Decompresses a `.emps` file to a temporary MPS
///       file using FFI.
///     - `get_lp(&self, name: &str)`: Downloads, decompresses, and parses a Netlib LP case into an
///       `mps::model::Model<f32>`.
///
/// # Safety
/// This module uses unsafe code to interact with C libraries and manipulate raw pointers.
/// Care must be taken to ensure correct usage of FFI and resource management.
///
/// # Tests
/// Includes parameterized tests to verify downloading of all supported Netlib cases.
///
/// # Dependencies
/// - `libc`: For FFI and C file operations.
/// - `reqwest`: For HTTP downloads.
/// - `tempfile`: For temporary file management.
/// - `mps`: For parsing MPS files.
///
/// # Example
/// ```ignore
/// let loader = NetlibLoader::new();
/// let model = loader.get_lp("afiro").unwrap();
/// ```
use libc;
use problemo::Problem;
use problemo::common::{GlossProblemResult, IntoCommonProblem};
use serde::Deserialize;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use tempfile::NamedTempFile;

use crate::{get_cache_dir, get_data_dir};

unsafe extern "C" {
    pub fn set_emps_output(f: *mut libc::FILE);
    pub fn emps_init();
    pub fn process_from_filename(filename: *mut libc::c_char);
}

/// The C code (emps.c) uses global variables and is not reentrant.
static EMPS_LOCK: Mutex<()> = Mutex::new(());

pub static URL: &str = "https://netlib.org/lp/data/";

#[derive(Debug, Deserialize)]
pub struct Record {
    name: String,
    rows: usize,
    columns: usize,
    nonzeros: usize,
    bytes: usize,
    bound_types: Option<String>,
    optimal_value: Option<f64>,
}

pub static NETLIB_CASE_DATA: LazyLock<HashMap<String, Record>> = LazyLock::new(|| {
    let mut case_data = HashMap::new();
    let csv_path = format!("{}/netlib.csv", get_data_dir());
    let mut rdr = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .from_path(&csv_path)
        .unwrap();
    for results in rdr.deserialize() {
        let record: Record = results.unwrap();
        case_data.insert(record.name.to_lowercase(), record);
    }
    case_data
});

fn get_internal_name(name: &str) -> String {
    name.replace('.', "_").replace('-', "_")
}

fn download_compressed(name: &str) -> Result<PathBuf, Problem> {
    let cache_dir = get_cache_dir();

    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

    if !NETLIB_CASE_DATA.contains_key(name) {
        return Err(format!("Unknown Netlib case: {}", name).gloss());
    }

    // Download file if it does not exist
    let internal_name = get_internal_name(name);
    let cached_path = Path::new(&format!("{}/{}.emps", &cache_dir, internal_name)).to_owned();
    if !Path::new(&format!("{}/{}.emps", &cache_dir, internal_name)).exists() {
        let url = format!("{}{}", URL, name);
        let response = reqwest::blocking::get(&url)
            .map_err(|e| format!("Failed to download file: {}", e).gloss())?;
        if !response.status().is_success() {
            return Err(format!("HTTP error: {} {}", response.status(), name).gloss());
        }
        let bytes = response
            .bytes()
            .map_err(|e| format!("Failed to read response bytes: {}", e).gloss())?;

        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(cached_path.to_str().unwrap())
            .expect("Failed to create file");
        file.write_all(&bytes).expect("Unable to write file.");
        file.sync_all().expect("Failed to sync file");
    }

    Ok(cached_path)
}

fn decompress_mps(emps_path: &str) -> Result<NamedTempFile, Problem> {
    if !Path::new(&emps_path).exists() {
        Err(format!("EMPS file does not exist: {}", emps_path).gloss())?;
    }

    let infile1 = CString::new(emps_path).unwrap();
    let infile1_ptr = infile1.into_raw();

    let tmpfile = NamedTempFile::new().expect("Failed to create temporary file");
    let out_path = CString::new(tmpfile.path().to_str().unwrap()).unwrap();
    let out_mode = CString::new("w").unwrap();

    // Serialize access — the C code uses global state and is not reentrant.
    let _guard = EMPS_LOCK.lock().unwrap();

    unsafe {
        let out_file = libc::fopen(out_path.as_ptr(), out_mode.as_ptr());
        if out_file.is_null() {
            Err("Failed to open output file for EMPS decompression".to_string()).gloss()?;
        }

        set_emps_output(out_file);
        emps_init();
        process_from_filename(infile1_ptr);

        libc::fflush(out_file);
        libc::fclose(out_file);
    }

    Ok(tmpfile)
}

#[allow(unused)]
pub struct MPSCase {
    model: mps::model::Model<f32>,
    name: String,
    rows: usize,
    columns: usize,
    nonzeros: usize,
    bytes: usize,
    bound_types: Option<String>,
    optimal_value: Option<f64>,
}

impl MPSCase {
    pub fn model(&self) -> &mps::model::Model<f32> {
        &self.model
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn nonzeros(&self) -> usize {
        self.nonzeros
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn bound_types(&self) -> Option<&String> {
        self.bound_types.as_ref()
    }

    pub fn optimal_value(&self) -> Option<f64> {
        self.optimal_value
    }
}

pub fn get_case(name: &str) -> Result<MPSCase, Problem> {
    let path = download_compressed(&name.to_lowercase())?;

    // Convert name to internal name and get the data
    let emps_path = path.to_str().unwrap();
    let mps_file = decompress_mps(&emps_path)
        .map_err(|e| format!("Unable to decompress emps file: {}", e).gloss())?;

    let mut contents = String::new();
    let mut file = mps_file.reopen().unwrap(); // This returns a File
    file.read_to_string(&mut contents).unwrap();

    let mps_parser = mps::Parser::<f32>::parse(&contents)
        .map_err(|e| format!("Unable to parse mps file: {}", e).gloss())?;
    let mps_model: mps::model::Model<f32> = mps_parser
        .try_into()
        .map_err(|e| format!("Failed to convert MPS model: {}", e).gloss())?;

    // Get metadata from CSV
    let Record {
        rows,
        columns,
        nonzeros,
        bytes,
        bound_types,
        optimal_value,
        ..
    } = NETLIB_CASE_DATA
        .get(name)
        .ok_or_else(|| "Unable to find metadata for case".gloss())?;

    Ok(MPSCase {
        model: mps_model,
        name: name.to_string(),
        rows: *rows,
        columns: *columns,
        nonzeros: *nonzeros,
        bytes: *bytes,
        bound_types: bound_types.clone(),
        optimal_value: *optimal_value,
    })
}

pub fn get_n_vars(name: &str) -> Result<usize, Problem> {
    NETLIB_CASE_DATA
        .get(name)
        .map(|record| record.columns)
        .ok_or_else(|| "Unable to find metadata for case".gloss())
}

pub fn get_n_constraints(name: &str) -> Result<usize, Problem> {
    NETLIB_CASE_DATA
        .get(name)
        .map(|record| record.rows)
        .ok_or_else(|| "Unable to find metadata for case".gloss())
}

pub fn get_n_nonzeros(name: &str) -> Result<usize, Problem> {
    NETLIB_CASE_DATA
        .get(name)
        .map(|record| record.nonzeros)
        .ok_or_else(|| "Unable to find metadata for case".gloss())
}

pub fn get_n_bytes(name: &str) -> Result<usize, Problem> {
    NETLIB_CASE_DATA
        .get(name)
        .map(|record| record.bytes)
        .ok_or_else(|| "Unable to find metadata for case".gloss())
}

pub fn get_bound_types(name: &str) -> Option<&String> {
    NETLIB_CASE_DATA
        .get(name)
        .and_then(|record| record.bound_types.as_ref())
}

pub fn get_optimal_value(name: &str) -> Option<f64> {
    NETLIB_CASE_DATA
        .get(name)
        .and_then(|record| record.optimal_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[value_parameterized_test(
      values = [
          "25fv47",
          "adlittle",
          "afiro",
          "agg",
          "agg2",
          "agg3",
          "bandm",
          "beaconfd",
          "blend",
          "bnl1",
          "bnl2",
          "boeing1",
          "boeing2",
          "bore3d",
          "brandy",
          "capri",
          "cycle",
          "czprob",
          "d2q06c",
          "d6cube",
          "degen2",
          "degen3",
          "dfl001",
          "e226",
          "etamacro",
          "fffff800",
          "finnis",
          "fit1d",
          "fit1p",
          "fit2d",
          "forplan",
          "ganges",
          "gfrd-pnc",
          "greenbea",
          "greenbeb",
          "grow15",
          "grow22",
          "grow7",
          "israel",
          "kb2",
          "lotfi",
          "maros-r7",
          "maros",
          "modszk1",
          "nesm",
          "perold",
          "pilot.ja",
          "pilot.we",
          "pilot",
          "pilot4",
          "pilot87",
          "pilotnov",
          "recipe",
          "sc105",
          "sc205",
          "sc50a",
          "sc50b",
          "scagr25",
          "scagr7",
          "scfxm1",
          "scfxm2",
          "scfxm3",
          "scorpion",
          "scrs8",
          "scsd1",
          "scsd6",
          "scsd8",
          "sctap1",
          "sctap2",
          "sctap3",
          "seba",
          "share1b",
          "share2b",
          "shell",
          "ship04l",
          "ship04s",
          "ship08l",
          "ship08s",
          "ship12l",
          "ship12s",
          "sierra",
          "stair",
          "standata",
          "standgub",
          "standmps",
          "stocfor1",
          "stocfor2",
          "tuff",
          "vtp.base",
          "wood1p",
          "woodw",
      ]
    )]
    #[allow(non_snake_case)]
    fn test_get_case(name: &str) {
        let mps_case = get_case(name).unwrap();
    }
}

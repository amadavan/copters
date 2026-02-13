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
use std::collections::HashSet;
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use tempfile::NamedTempFile;

unsafe extern "C" {
    pub fn set_emps_output(f: *mut libc::FILE);
    pub fn emps_init();
    pub fn process_from_filename(filename: *mut libc::c_char);
}

/// The C code (emps.c) uses global variables and is not reentrant.
static EMPS_LOCK: Mutex<()> = Mutex::new(());

pub static URL: &str = "https://netlib.org/lp/data/";

pub static NETLIB_CASES: LazyLock<HashSet<String>> = LazyLock::new(|| {
    let cases = vec![
        "agg", "ship04l", "d2q06c", "e226", "25fv47", "bore3d", "ganges", "adlittle", "forplan",
        "sc205", "scrs8", "wood1p", "boeing1", "kb2", "ship08s", "scfxm1", "agg2", "finnis",
        "dfl001", "pilot87", "sctap1", "agg3", "grow7", "scorpion", "maros", "shell", "greenbeb",
        "sc50b", "recipe", "sierra", "scagr25", "modszk1", "ship12l", "stair", "cycle", "sc105",
        "pilot_ja", "beaconfd", "czprob", "pilot_we", "standgub", "standmps", "scsd8", "woodw",
        "scsd6", "scsd1", "share2b", "gfrd_pnc", "bnl2", "stocfor2", "nesm", "share1b", "ship04s",
        "grow15", "maros_r7", "blend", "lotfi", "standata", "d6cube", "degen3", "capri", "grow22",
        "etamacro", "ship08l", "afiro", "degen2", "boeing2", "fit1d", "scfxm2", "sctap3", "fit1p",
        "pilot", "fit2d", "bandm", "sctap2", "scfxm3", "brandy", "greenbea", "tuff", "sc50a",
        "vtp_base", "pilotnov", "ship12s", "seba", "fffff800", "bnl1", "scagr7", "stocfor1",
        "perold",
    ];

    HashSet::from_iter(cases.iter().map(|s| s.to_string()))
});

fn get_cache_dir() -> String {
    format!("{}/artifacts", env!("CARGO_MANIFEST_DIR"))
}

pub fn download_compressed(name: &str) -> Result<PathBuf, Problem> {
    let cache_dir = get_cache_dir();

    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

    if !NETLIB_CASES.contains(name) {
        return Err(format!("Unknown Netlib case: {}", name).gloss());
    }

    // Download file if it does not exist
    if !Path::new(&format!("{}/{}.emps", &cache_dir, name)).exists() {
        let url = format!("{}{}", URL, name);
        let response = reqwest::blocking::get(&url)
            .map_err(|e| format!("Failed to download file: {}", e).gloss())?;
        if !response.status().is_success() {
            return Err(format!("HTTP error: {} {}", response.status(), name).gloss());
        }
        let bytes = response
            .bytes()
            .map_err(|e| format!("Failed to read response bytes: {}", e).gloss())?;

        let file_name = format!("{}/{}.emps", &cache_dir, name);

        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_name)
            .expect("Failed to create file");
        file.write_all(&bytes).expect("Unable to write file.");
        file.sync_all().expect("Failed to sync file");
    }

    Ok(Path::new(&format!("{}/{}.emps", &cache_dir, name)).to_owned())
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

pub fn get_lp(name: &str) -> Result<mps::model::Model<f32>, Problem> {
    download_compressed(name)?;
    let emps_path = format!("{}/{}.emps", get_cache_dir(), name);
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
    Ok(mps_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[value_parameterized_test(
      values = [
          "israel",
          "scagr7",
          "ship08s",
          "vtp_base",
          "bnl1",
          "pilot",
          "standgub",
          "scsd1",
          "sc205",
          "adlittle",
          "ship04s",
          "scfxm2",
          "agg",
          "agg3",
          "scorpion",
          "shell",
          "greenbeb",
          "fit2d",
          "bandm",
          "share2b",
          "sc105",
          "nesm",
          "boeing2",
          "sc50b",
          "scfxm3",
          "stair",
          "stocfor1",
          "maros",
          "bore3d",
          "scsd8",
          "stocfor2",
          "25fv47",
          "sctap1",
          "ship12l",
          "beaconfd",
          "modszk1",
          "cycle",
          "ship12s",
          "forplan",
          "kb2",
          "recipe",
          "fit1d",
          "e226",
          "etamacro",
          "perold",
          "fffff800",
          "sierra",
          "maros_r7",
          "tuff",
          "pilotnov",
          "dfl001",
          "pilot87",
          "pilot_we",
          "capri",
          "pilot4",
          "wood1p",
          "woodw",
          "ship04l",
          "grow15",
          "degen3",
          "fit1p",
          "standata",
          "greenbea",
          "czprob",
          "scfxm1",
          "sc50a",
          "agg2",
          "standmps",
          "share1b",
          "afiro",
          "seba",
          "degen2",
          "scagr25",
          "scrs8",
          "ganges",
          "brandy",
          "scsd6",
          "boeing1",
          "grow7",
          "bnl2",
          "sctap3",
          "pilot_ja",
          "blend",
          "sctap2",
          "d6cube",
          "grow22",
          "gfrd_pnc",
          "ship08l",
          "d2q06c",
          "lotfi",
          "finnis"
      ]
    )]
    #[allow(non_snake_case)]
    fn test_download_compressed(name: &str) {
        NetlibLoader::download_compressed(name).unwrap();
    }

    #[value_parameterized_test(
      values = [
          "israel",
          "scagr7",
          "ship08s",
          "vtp_base",
          "bnl1",
          "pilot",
          "standgub",
          "scsd1",
          "sc205",
          "adlittle",
          "ship04s",
          "scfxm2",
          "agg",
          "agg3",
          "scorpion",
          "shell",
          "greenbeb",
          "fit2d",
          "bandm",
          "share2b",
          "sc105",
          "nesm",
          "boeing2",
          "sc50b",
          "scfxm3",
          "stair",
          "stocfor1",
          "maros",
          "bore3d",
          "scsd8",
          "stocfor2",
          "25fv47",
          "sctap1",
          "ship12l",
          "beaconfd",
          "modszk1",
          "cycle",
          "ship12s",
          "forplan",
          "kb2",
          "recipe",
          "fit1d",
          "e226",
          "etamacro",
          "perold",
          "fffff800",
          "sierra",
          "maros_r7",
          "tuff",
          "pilotnov",
          "dfl001",
          "pilot87",
          "pilot_we",
          "capri",
          "pilot4",
          "wood1p",
          "woodw",
          "ship04l",
          "grow15",
          "degen3",
          "fit1p",
          "standata",
          "greenbea",
          "czprob",
          "scfxm1",
          "sc50a",
          "agg2",
          "standmps",
          "share1b",
          "afiro",
          "seba",
          "degen2",
          "scagr25",
          "scrs8",
          "ganges",
          "brandy",
          "scsd6",
          "boeing1",
          "grow7",
          "bnl2",
          "sctap3",
          "pilot_ja",
          "blend",
          "sctap2",
          "d6cube",
          "grow22",
          "gfrd_pnc",
          "ship08l",
          "d2q06c",
          "lotfi",
          "finnis"
      ]
    )]
    #[allow(non_snake_case)]
    fn test_get_lp(name: &str) {
        let model = NetlibLoader::get_lp(name).unwrap();
        // assert!(!model.rows.is_empty(), "Model has no rows");
        // assert!(!model.columns.is_empty(), "Model has no columns");
    }
}

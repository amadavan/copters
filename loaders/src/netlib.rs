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
use std::collections::HashMap;
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

pub static NETLIB_CASES: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("agg".to_string(), "agg".to_string());
    m.insert("ship04l".to_string(), "ship04l".to_string());
    m.insert("d2q06c".to_string(), "d2q06c".to_string());
    m.insert("e226".to_string(), "e226".to_string());
    m.insert("25fv47".to_string(), "25fv47".to_string());
    m.insert("bore3d".to_string(), "bore3d".to_string());
    m.insert("ganges".to_string(), "ganges".to_string());
    m.insert("adlittle".to_string(), "adlittle".to_string());
    m.insert("forplan".to_string(), "forplan".to_string());
    m.insert("sc205".to_string(), "sc205".to_string());
    m.insert("scrs8".to_string(), "scrs8".to_string());
    m.insert("wood1p".to_string(), "wood1p".to_string());
    m.insert("boeing1".to_string(), "boeing1".to_string());
    m.insert("kb2".to_string(), "kb2".to_string());
    m.insert("ship08s".to_string(), "ship08s".to_string());
    m.insert("scfxm1".to_string(), "scfxm1".to_string());
    m.insert("agg2".to_string(), "agg2".to_string());
    m.insert("finnis".to_string(), "finnis".to_string());
    m.insert("dfl001".to_string(), "dfl001".to_string());
    m.insert("pilot87".to_string(), "pilot87".to_string());
    m.insert("sctap1".to_string(), "sctap1".to_string());
    m.insert("agg3".to_string(), "agg3".to_string());
    m.insert("grow7".to_string(), "grow7".to_string());
    m.insert("scorpion".to_string(), "scorpion".to_string());
    m.insert("maros".to_string(), "maros".to_string());
    m.insert("shell".to_string(), "shell".to_string());
    m.insert("greenbeb".to_string(), "greenbeb".to_string());
    m.insert("sc50b".to_string(), "sc50b".to_string());
    m.insert("recipe".to_string(), "recipe".to_string());
    m.insert("sierra".to_string(), "sierra".to_string());
    m.insert("scagr25".to_string(), "scagr25".to_string());
    m.insert("modszk1".to_string(), "modszk1".to_string());
    m.insert("ship12l".to_string(), "ship12l".to_string());
    m.insert("stair".to_string(), "stair".to_string());
    m.insert("cycle".to_string(), "cycle".to_string());
    m.insert("sc105".to_string(), "sc105".to_string());
    m.insert("pilot_ja".to_string(), "pilot.ja".to_string());
    m.insert("beaconfd".to_string(), "beaconfd".to_string());
    m.insert("czprob".to_string(), "czprob".to_string());
    m.insert("pilot_we".to_string(), "pilot.we".to_string());
    m.insert("standgub".to_string(), "standgub".to_string());
    m.insert("standmps".to_string(), "standmps".to_string());
    m.insert("scsd8".to_string(), "scsd8".to_string());
    m.insert("woodw".to_string(), "woodw".to_string());
    m.insert("scsd6".to_string(), "scsd6".to_string());
    m.insert("scsd1".to_string(), "scsd1".to_string());
    m.insert("share2b".to_string(), "share2b".to_string());
    m.insert("gfrd_pnc".to_string(), "gfrd-pnc".to_string());
    m.insert("bnl2".to_string(), "bnl2".to_string());
    m.insert("stocfor2".to_string(), "stocfor2".to_string());
    m.insert("nesm".to_string(), "nesm".to_string());
    m.insert("share1b".to_string(), "share1b".to_string());
    m.insert("ship04s".to_string(), "ship04s".to_string());
    m.insert("grow15".to_string(), "grow15".to_string());
    m.insert("maros_r7".to_string(), "maros-r7".to_string());
    m.insert("blend".to_string(), "blend".to_string());
    m.insert("lotfi".to_string(), "lotfi".to_string());
    m.insert("standata".to_string(), "standata".to_string());
    m.insert("d6cube".to_string(), "d6cube".to_string());
    m.insert("degen3".to_string(), "degen3".to_string());
    m.insert("capri".to_string(), "capri".to_string());
    m.insert("grow22".to_string(), "grow22".to_string());
    m.insert("etamacro".to_string(), "etamacro".to_string());
    m.insert("ship08l".to_string(), "ship08l".to_string());
    m.insert("afiro".to_string(), "afiro".to_string());
    m.insert("degen2".to_string(), "degen2".to_string());
    m.insert("boeing2".to_string(), "boeing2".to_string());
    m.insert("fit1d".to_string(), "fit1d".to_string());
    m.insert("scfxm2".to_string(), "scfxm2".to_string());
    m.insert("sctap3".to_string(), "sctap3".to_string());
    m.insert("fit1p".to_string(), "fit1p".to_string());
    m.insert("pilot".to_string(), "pilot".to_string());
    m.insert("fit2d".to_string(), "fit2d".to_string());
    m.insert("sctap2".to_string(), "sctap2".to_string());
    m.insert("scfxm3".to_string(), "scfxm3".to_string());
    m.insert("brandy".to_string(), "brandy".to_string());
    m.insert("greenbea".to_string(), "greenbea".to_string());
    m.insert("tuff".to_string(), "tuff".to_string());
    m.insert("sc50a".to_string(), "sc50a".to_string());
    m.insert("vtp_base".to_string(), "vtp.base".to_string());
    m.insert("pilotnov".to_string(), "pilotnov".to_string());
    m.insert("ship12s".to_string(), "ship12s".to_string());
    m.insert("seba".to_string(), "seba".to_string());
    m.insert("fffff800".to_string(), "fffff800".to_string());
    m.insert("israel".to_string(), "israel".to_string());
    m.insert("perold".to_string(), "perold".to_string());
    m.insert("pilot4".to_string(), "pilot4".to_string());
    m.insert("scagr7".to_string(), "scagr7".to_string());
    m.insert("bandm".to_string(), "bandm".to_string());
    m.insert("bnl1".to_string(), "bnl1".to_string());
    m.insert("stocfor1".to_string(), "stocfor1".to_string());
    m
});

fn get_cache_dir() -> String {
    format!("{}/artifacts", env!("CARGO_MANIFEST_DIR"))
}

pub fn download_compressed(name: &str) -> Result<PathBuf, Problem> {
    let cache_dir = get_cache_dir();

    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

    if !NETLIB_CASES.contains_key(name) {
        return Err(format!("Unknown Netlib case: {}", name).gloss());
    }

    // Download file if it does not exist
    if !Path::new(&format!("{}/{}.emps", &cache_dir, name)).exists() {
        let url = format!("{}{}", URL, NETLIB_CASES.get(name).unwrap());
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

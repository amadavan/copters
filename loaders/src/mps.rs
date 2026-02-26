use libc;
use problemo::Problem;
use problemo::common::{GlossProblemResult, IntoCommonProblem};
use std::ffi::CString;
use std::path::Path;
use std::sync::Mutex;
use tempfile::NamedTempFile;

unsafe extern "C" {
    pub fn set_emps_output(f: *mut libc::FILE);
    pub fn emps_init();
    pub fn process_from_filename(filename: *mut libc::c_char);
}

/// The C code (emps.c) uses global variables and is not reentrant.
static EMPS_LOCK: Mutex<()> = Mutex::new(());

pub fn decompress_mps(emps_path: &str) -> Result<NamedTempFile, Problem> {
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

#[cfg(test)]
mod test {
    use rstest::rstest;
    use std::{fs::OpenOptions, io::Write, path::PathBuf};

    use crate::get_cache_dir;

    use super::*;

    static URL: &str = "https://netlib.org/lp/data/";

    fn get_internal_name(name: &str) -> String {
        // name.replace(".", "_").to_lowercase()
        name.to_lowercase()
    }

    fn download_compressed(name: &str) -> Result<PathBuf, Problem> {
        let cache_dir = get_cache_dir() + "/emps";

        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        // if !NETLIB_CASE_DATA.contains_key(name) {
        //     return Err(format!("Unknown Netlib case: {}", name).gloss());
        // }

        // Download file if it does not exist
        let internal_name = get_internal_name(name);
        let cached_path = Path::new(&format!("{}/{}.emps", &cache_dir, internal_name)).to_owned();
        if !cached_path.exists() {
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

    #[rstest]
    fn test_decompress(
        #[values(
            "25fv47", "adlittle", "afiro", "agg", "agg2", "agg3", "bandm", "beaconfd", "blend",
            "bnl1", "bnl2", "boeing1", "boeing2", "bore3d", "brandy", "capri", "cycle", "czprob",
            "d2q06c", "d6cube", "degen2", "degen3", "dfl001", "e226", "etamacro", "fffff800",
            "finnis", "fit1d", "fit1p", "fit2d", "forplan", "ganges", "gfrd-pnc", "greenbea",
            "greenbeb", "grow15", "grow22", "grow7", "israel", "kb2", "lotfi", "maros-r7", "maros",
            "modszk1", "nesm", "perold", "pilot.ja", "pilot.we", "pilot", "pilot4", "pilot87",
            "pilotnov", "recipe", "sc105", "sc205", "sc50a", "sc50b", "scagr25", "scagr7",
            "scfxm1", "scfxm2", "scfxm3", "scorpion", "scrs8", "scsd1", "scsd6", "scsd8", "sctap1",
            "sctap2", "sctap3", "seba", "share1b", "share2b", "shell", "ship04l", "ship04s",
            "ship08l", "ship08s", "ship12l", "ship12s", "sierra", "stair", "standata", "standgub",
            "standmps", "stocfor1", "stocfor2", "tuff", "vtp.base", "wood1p", "woodw"
        )]
        case_name: &str,
    ) {
        // let case_name = case_name.to_uppercase();
        let path = download_compressed(&case_name).expect("Failed to download compressed MPS file");

        // Convert name to internal name and get the data
        let emps_path = path.to_str().unwrap();
        decompress_mps(&emps_path)
            .ok()
            .expect("Failed to decompress MPS file");
    }
}

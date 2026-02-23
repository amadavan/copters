use problemo::{Problem, common::IntoCommonProblem};
use sif_rs::SIF;
use std::{io::Read, path::Path};

static MAROS_MEZAROS_QP_TAR_URL: &str =
    "https://bitbucket.org/optrove/maros-meszaros/get/v0.1.tar.gz";
static NETLIB_LP_TAR_URL: &str = "https://bitbucket.org/optrove/netlib-lp/get/v0.1.tar.gz";

fn download_http(url: &str) -> Result<Vec<u8>, Problem> {
    let response =
        reqwest::blocking::get(url).map_err(|e| format!("HTTP request failed: {e}").gloss())?;
    let total = response.content_length().unwrap_or(0);
    let pb = indicatif::ProgressBar::new(total);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner} [{bar:40}] {bytes}/{total_bytes} ({eta})")
            .unwrap(),
    );

    let mut buf = Vec::new();
    pb.wrap_read(response)
        .read_to_end(&mut buf)
        .map_err(|e| format!("HTTP read failed: {e}").gloss())?;
    pb.finish_with_message(format!("Downloaded {url}"));
    Ok(buf)
}

fn unpack_optrove(tar_gz: &[u8], target_dir: String) -> Result<(), Problem> {
    let tar = flate2::read::GzDecoder::new(&tar_gz[..]);
    let mut archive = tar::Archive::new(tar);
    std::fs::create_dir_all(&target_dir)?;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;

        if path.extension().and_then(|s| s.to_str()) == Some("SIF") {
            if let Some(filename) = path.file_name() {
                entry.unpack(Path::new(&target_dir).join(filename))?;
            }
        }
    }
    Ok(())
}

#[allow(unused)]
fn download_maros_mezaros_qp() -> Result<(), Problem> {
    let filename = "marosmezaros.tar.gz";
    if !Path::new(&format!("{}/{}", crate::get_cache_dir(), filename)).exists() {
        // Download the tar file
        let tar_gz = download_http(MAROS_MEZAROS_QP_TAR_URL)?;
        std::fs::create_dir_all(format!("{}", crate::get_cache_dir()))?;
        std::fs::write(format!("{}/{}", crate::get_cache_dir(), filename), &tar_gz)?;
    }
    let tar_gz = std::fs::read(format!("{}/{}", crate::get_cache_dir(), filename))?;

    let target_dir = format!("{}/{}", crate::get_cache_dir(), "maros_mezaros");
    std::fs::create_dir_all(&target_dir)?;
    unpack_optrove(&tar_gz, target_dir)?;

    Ok(())
}

#[allow(unused)]
fn download_netlib_lp() -> Result<(), Problem> {
    let filename = "netlib.tar.gz";
    if !Path::new(&format!("{}/{}", crate::get_cache_dir(), filename)).exists() {
        // Download the tar file
        let tar_gz = download_http(NETLIB_LP_TAR_URL)?;
        std::fs::create_dir_all(format!("{}", crate::get_cache_dir()))?;
        std::fs::write(format!("{}/{}", crate::get_cache_dir(), filename), &tar_gz)?;
    }
    let tar_gz = std::fs::read(format!("{}/{}", crate::get_cache_dir(), filename))?;

    let target_dir = format!("{}/{}", crate::get_cache_dir(), "netlib");
    std::fs::create_dir_all(&target_dir)?;
    unpack_optrove(&tar_gz, target_dir)?;

    Ok(())
}

pub mod netlib {
    use super::*;

    pub fn get_case(case_name: &str) -> Result<SIF, Problem> {
        let file_path = format!(
            "{}/netlib/{}.SIF",
            crate::get_cache_dir(),
            case_name.to_uppercase()
        );
        if !Path::new(&file_path).exists() {
            download_netlib_lp()?;
        }
        if !Path::new(&file_path).exists() {
            return Err(format!(
                "SIF file for case '{}' not found at '{}'",
                case_name, file_path
            )
            .gloss());
        }
        let sif_data = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read SIF file '{}': {e}", file_path).gloss())?;
        sif_rs::parse_sif(&sif_data).map_err(|_| "Unable to parse SIF file".gloss())
    }
}

pub mod maros_mezaros {
    use super::*;

    pub fn get_case(case_name: &str) -> Result<SIF, Problem> {
        let file_path = format!(
            "{}/maros_mezaros/{}.SIF",
            crate::get_cache_dir(),
            case_name.to_uppercase()
        );
        if !Path::new(&file_path).exists() {
            download_maros_mezaros_qp()?;
        }
        if !Path::new(&file_path).exists() {
            return Err(format!(
                "SIF file for case '{}' not found at '{}'",
                case_name, file_path
            )
            .gloss());
        }
        let sif_data = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read SIF file '{}': {e}", file_path).gloss())?;
        sif_rs::parse_sif(&sif_data).map_err(|_| "Unable to parse SIF file".gloss())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;
    use rstest_reuse::{apply, template};

    #[template]
    #[rstest]
    pub fn maros_mezaros_cases(
        #[values(
            "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP", "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP",
            "BOYD1", "BOYD2", "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201",
            "CONT-300", "CVXQP1_L", "CVXQP1_M", "CVXQP1_S", "CVXQP2_L", "CVXQP2_M", "CVXQP2_S",
            "CVXQP3_L", "CVXQP3_M", "CVXQP3_S", "DPKLO1", "DTOC3", "DUAL1", "DUAL2", "DUAL3",
            "DUAL4", "DUALC1", "DUALC2", "DUALC5", "DUALC8", "EXDATA", "GENHS28", "GOULDQP2",
            "GOULDQP3", "HS118", "HS21", "HS268", "HS35", "HS35MOD", "HS51", "HS52", "HS53",
            "HS76", "HUES-MOD", "HUESTIS", "KSIP", "LASER", "LISWET1", "LISWET10", "LISWET11",
            "LISWET12", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6", "LISWET7",
            "LISWET8", "LISWET9", "LOTSCHD", "MOSARQP1", "MOSARQP2", "POWELL20", "PRIMAL1",
            "PRIMAL2", "PRIMAL3", "PRIMAL4", "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8",
            "Q25FV47", "QADLITTL", "QAFIRO", "QBANDM", "QBEACONF", "QBORE3D", "QBRANDY", "QCAPRI",
            "QE226", "QETAMACR", "QFFFFF80", "QFORPLAN", "QGFRDXPN", "QGROW15", "QGROW22",
            "QGROW7", "QISRAEL", "QPCBLEND", "QPCBOEI1", "QPCBOEI2", "QPCSTAIR", "QPILOTNO",
            "QPTEST", "QRECIPE", "QSC205", "QSCAGR25", "QSCAGR7", "QSCFXM1", "QSCFXM2", "QSCFXM3",
            "QSCORPIO", "QSCRS8", "QSCSD1", "QSCSD6", "QSCSD8", "QSCTAP1", "QSCTAP2", "QSCTAP3",
            "QSEBA", "QSHARE1B", "QSHARE2B", "QSHELL", "QSHIP04L", "QSHIP04S", "QSHIP08L",
            "QSHIP08S", "QSHIP12L", "QSHIP12S", "QSIERRA", "QSTAIR", "QSTANDAT", "S268", "STADAT1",
            "STADAT2", "STADAT3", "STCQP1", "STCQP2", "TAME", "UBH1", "VALUES", "YAO", "ZECEVIC2"
        )]
        case_name: &str,
    ) {
    }

    #[template]
    #[rstest]
    pub fn netlib_cases(
        #[values(
            "25fv47", "80bau3b", "adlittle", "afiro", "agg", "agg2", "agg3", "bandm", "beaconfd",
            "blend", "bnl1", "bnl2", "boeing1", "boeing2", "bore3d", "brandy", "capri", "cre-a",
            "cre-b", "cre-c", "cre-d", "cycle", "czprob", "d2q06c", "d6cube", "degen2", "degen3",
            "dfl001", "e226", "etamacro", "fffff800", "finnis", "fit1d", "fit1p", "fit2d", "fit2p",
            "forplan", "ganges", "gfrd-pnc", "greenbea", "greenbeb", "grow15", "grow22", "grow7",
            "israel", "kb2", "ken-07", "ken-11", "ken-13", "ken-18", "lotfi", "maros-r7", "maros",
            "modszk1", "nesm", "osa-07", "osa-14", "osa-30", "osa-60", "pds-02", "pds-06",
            "pds-10", "pds-20", "perold", "pilot-ja", "pilot-we", "pilot", "pilot4", "pilot87",
            "pilotnov", "qap8", "qap12", "qap15", "recipelp", "sc105", "sc205", "sc50a", "sc50b",
            "scagr25", "scagr7", "scfxm1", "scfxm2", "scfxm3", "scorpion", "scrs8", "scsd1",
            "scsd6", "scsd8", "sctap1", "sctap2", "sctap3", "seba", "share1b", "share2b", "shell",
            "ship04l", "ship04s", "ship08l", "ship08s", "ship12l", "ship12s", "sierra", "stair",
            "standata", "standgub", "standmps", "stocfor1", "stocfor2", "tuff", "vtp-base",
            "wood1p", "woodw"
        )]
        case_name: &str,
    ) {
    }

    #[apply(maros_mezaros_cases)]
    fn get_maros_mezaros_qp(case_name: &str) {
        let case_name = case_name.to_uppercase();
        let _ =
            maros_mezaros::get_case(&case_name).expect("Failed to get Maros-Mezaros QP dataset");

        // Verify that the expected files are present
        let maros_mezaros_dir = format!("{}/maros_mezaros", crate::get_cache_dir());
        let expected_file =
            std::path::Path::new(&maros_mezaros_dir).join(format!("{}.SIF", case_name));
        assert!(
            expected_file.exists(),
            "Maros-Mezaros case '{}' not found in cache",
            case_name
        );
    }

    #[apply(netlib_cases)]
    fn get_netlib_lp(case_name: &str) {
        let case_name = case_name.to_uppercase();
        netlib::get_case(&case_name).expect("Failed to get Netlib LP dataset");

        // Verify that the expected files are present
        let netlib_dir = format!("{}/netlib", crate::get_cache_dir());
        let expected_file = std::path::Path::new(&netlib_dir).join(format!("{}.SIF", case_name));
        assert!(
            expected_file.exists(),
            "Netlib case '{}' not found in cache",
            case_name
        );
    }
}

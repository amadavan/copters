use std::{collections::HashMap, sync::LazyLock};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct MetaData {
    name: String,
    rows: usize,
    columns: usize,
    nonzeros: usize,
    bytes: usize,
    bound_types: Option<String>,
    optimal_value: Option<f64>,
}

impl MetaData {
    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_rows(&self) -> usize {
        self.rows
    }

    pub fn get_columns(&self) -> usize {
        self.columns
    }

    pub fn get_nonzeros(&self) -> usize {
        self.nonzeros
    }

    pub fn get_bytes(&self) -> usize {
        self.bytes
    }

    pub fn get_bound_types(&self) -> Option<&str> {
        self.bound_types.as_deref()
    }

    pub fn get_optimal_value(&self) -> Option<f64> {
        self.optimal_value
    }
}

pub static NETLIB_CASE_DATA: LazyLock<HashMap<String, MetaData>> = LazyLock::new(|| {
    let mut case_data = HashMap::new();
    let csv_path = format!("{}/netlib.csv", crate::get_data_dir());
    let mut rdr = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .from_path(&csv_path)
        .unwrap();
    for results in rdr.deserialize() {
        let record: MetaData = results.unwrap();
        case_data.insert(record.name.to_lowercase(), record);
    }
    case_data
});

pub fn get_case_metadata(case_name: &str) -> Option<&MetaData> {
    NETLIB_CASE_DATA.get(&case_name.to_lowercase())
}

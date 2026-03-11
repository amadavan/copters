pub const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data");
pub const CACHE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/artifacts");

pub fn get_data_dir() -> &'static str {
    DATA_DIR
}

pub fn get_cache_dir() -> &'static str {
    CACHE_DIR
}

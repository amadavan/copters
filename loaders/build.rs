fn main() {
    cc::Build::new()
        .file("src/emps.c")
        .define("BUILDING_FOR_RUST", None)
        .warnings(false) // disables -Wall
        .flag_if_supported("-w") // disables all warnings for gcc/clang
        .compile("emps");
}

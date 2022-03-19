use std::env;

fn main() {
    let path = env::current_dir().unwrap();
    println!("cargo:rustc-link-search={}/libblis", path.display());
    println!("cargo:rustc-link-lib=static=blis");
}

fn main() {
    println!("cargo:rustc-link-search={}/libblis", "/home/cadenmiller/Documents/coding/neuroevolution");
    println!("cargo:rustc-link-lib=static=blis");
}

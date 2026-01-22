use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::io::Write;
use std::path::PathBuf;

use reqwest::blocking;

const CACHE_NAME: &str = ".cache/cellcast/weights";

/// TODO
pub fn fetch_weights(url: &str, file_name: &str, verbose: bool) -> Result<PathBuf, Box<dyn Error>> {
    let cache_dir = get_cache_dir()?;
    let weights_path = cache_dir.join(file_name);
    if weights_path.exists() {
        if verbose {
            println!("[INFO] Using cached weights at: {}", cache_dir.display());
        }
    } else {
        if verbose {
            println!("[INFO] Saving weights to: {}", cache_dir.display());
        }
        _ = download_weights(url, &weights_path, verbose);
    }

    Ok(weights_path)
}

/// Download a file
fn download_weights(url: &str, file_path: &PathBuf, verbose: bool) -> Result<(), Box<dyn Error>> {
    if verbose {
        println!("[INFO] Downloading weights from: {}", url);
    }
    let response = blocking::get(url).expect(&format!("Failed to get a response from {}.", url));
    let bytes = response.bytes()?;
    let mut weights = fs::File::create(&file_path)?;
    weights.write_all(&bytes)?;

    Ok(())
}

/// Get temporary cache directory for the system
fn get_cache_dir() -> io::Result<PathBuf> {
    let dir = env::home_dir().unwrap_or_else(|| env::temp_dir());
    let dir = dir.join(CACHE_NAME);
    fs::create_dir_all(&dir)?;

    Ok(dir)
}

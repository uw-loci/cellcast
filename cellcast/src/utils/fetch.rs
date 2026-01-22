use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::io::Write;
use std::path::PathBuf;

use reqwest::blocking;

const CACHE_NAME: &str = ".cache/cellcast/weights";

/// Fetch the requested weights.
///
/// # Description
///
/// Fetches the requested `.bin` weight file from the given `url`. If the
/// weights have already been downloaded and saved in the cache directory then
/// no download occurs and the cached weights are used.
///
/// # Arguments
///
/// * `url`: The URL to the `.bin` model weights.
/// * `file_name`: The name of the model weights file.
/// * `verbose`: If `true` then "INFO" status updates are printed to the
///    console. If `false`, then nothing is printed.
///
/// # Returns
///
/// * `Ok(PathBuf)`: The validate path to the `.bin` weights file.
/// * `Err(Error)`: If the `url` is not valid. If the `file_name` is not valid.
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

/// Download and save a .bin weights file.
///
/// # Arguments
///
/// * `url`: The URL to the `.bin` model weights.
/// * `file_path`: The file path to the cache directory where the weights are to
///   be saved.
/// * `verbose`: If `true` then "INFO" status updates are printed to the
///    console. If `false`, then nothing is printed.
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

/// Get the weights cache directory.
fn get_cache_dir() -> io::Result<PathBuf> {
    let dir = env::home_dir().unwrap_or_else(|| env::temp_dir());
    let dir = dir.join(CACHE_NAME);
    fs::create_dir_all(&dir)?;

    Ok(dir)
}

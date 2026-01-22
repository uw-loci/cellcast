use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::io::Write;
use std::path::PathBuf;

use reqwest::blocking;

/// TODO
pub fn fetch_weights(url: &str, file_name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let cache_dir = get_cache_dir()?;
    let weights_path = cache_dir.join(file_name);
    if weights_path.exists() {
        println!(
            "[DEBUG] using cached weights at: {}",
            weights_path.display()
        );
    } else {
        println!("[DEBUG] downloading weights to: {}", weights_path.display());
        _ = download_weights(url, file_name, &weights_path);
    }

    Ok(weights_path)
}

/// Download a file
fn download_weights(url: &str, file_name: &str, file_path: &PathBuf) -> Result<(), Box<dyn Error>> {
    println!("[DEBUG] downloading weights from: {}", url);
    let response = blocking::get(url)?;
    if !response.status().is_success() {
        return Err(format!(
            "Failed to download {} weights from {}.\n{}",
            file_name,
            url,
            response.status()
        )
        .into());
    }
    let bytes = response.bytes()?;
    let mut weights = fs::File::create(&file_path)?;
    weights.write_all(&bytes)?;
    println!("[DEBUG] weights downloaded to {}", file_path.display());

    Ok(())
}

/// Get temporary cache directory for the system
fn get_cache_dir() -> io::Result<PathBuf> {
    // TODO detect the system type and set "permenant" cache directory appropriately
    let dir = env::temp_dir().join("cellcast_weights");
    fs::create_dir_all(&dir)?;

    Ok(dir)
}

use anyhow::Result;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let temp_path = unique_temp_path(path);

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temp_path)?;

    file.write_all(contents)?;
    file.sync_all()?;
    drop(file);

    if let Err(e) = fs::rename(&temp_path, path) {
        if path.exists() {
            let _ = fs::remove_file(path);
        }
        fs::rename(&temp_path, path).map_err(|_| e)?;
    }

    Ok(())
}

fn unique_temp_path(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.with_extension(format!("tmp.{}.{}", pid, nanos))
}

//! GPU memory probe.
//!
//! Returns the bytes practically available for the GPU working set plus the
//! device name, or `None` when no probeable GPU is present (pass `--budget`).

/// On unified-memory Macs, Metal's recommended working-set size is the
/// practical "VRAM" ceiling (~75% of RAM).
#[cfg(target_os = "macos")]
pub fn probe() -> Option<(u64, String)> {
    let device = metal::Device::system_default()?;
    Some((
        device.recommended_max_working_set_size(),
        device.name().to_string(),
    ))
}

/// Elsewhere, ask NVML for the first NVIDIA GPU's currently free VRAM.
/// Free (not total) is what a fit can actually claim once the desktop and
/// other processes have taken their share.
#[cfg(not(target_os = "macos"))]
pub fn probe() -> Option<(u64, String)> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let device = nvml.device_by_index(0).ok()?;
    let mem = device.memory_info().ok()?;
    let name = device.name().ok()?;
    Some((mem.free, name))
}

/// What the budget line should call the probed number.
pub fn probe_source() -> &'static str {
    if cfg!(target_os = "macos") {
        "Metal recommendedMaxWorkingSetSize"
    } else {
        "NVML free VRAM"
    }
}

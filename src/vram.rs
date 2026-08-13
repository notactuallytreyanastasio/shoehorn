//! GPU memory probe for Apple Silicon.

/// Bytes the Metal device recommends keeping resident, and the device name.
/// On unified-memory Macs this is the practical "VRAM" ceiling (~75% of RAM).
pub fn probe() -> Option<(u64, String)> {
    let device = metal::Device::system_default()?;
    Some((
        device.recommended_max_working_set_size(),
        device.name().to_string(),
    ))
}

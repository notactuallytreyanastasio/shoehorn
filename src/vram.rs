//! GPU memory probe.
//!
//! Returns the bytes practically available for the GPU working set, the
//! device name, and what the number means — or `None` when no probeable GPU
//! is present (pass `--budget`).

/// On unified-memory Macs, Metal's recommended working-set size is the
/// practical "VRAM" ceiling (~75% of RAM).
#[cfg(target_os = "macos")]
pub fn probe() -> Option<(u64, String, &'static str)> {
    let device = metal::Device::system_default()?;
    Some((
        device.recommended_max_working_set_size(),
        device.name().to_string(),
        "Metal recommendedMaxWorkingSetSize",
    ))
}

/// Elsewhere: NVIDIA via NVML first, then AMD via rocm-smi. Free (not
/// total) VRAM in both cases — what a fit can actually claim once the
/// desktop and other processes have taken their share.
#[cfg(not(target_os = "macos"))]
pub fn probe() -> Option<(u64, String, &'static str)> {
    nvml_probe().or_else(rocm_probe)
}

#[cfg(not(target_os = "macos"))]
fn nvml_probe() -> Option<(u64, String, &'static str)> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let device = nvml.device_by_index(0).ok()?;
    let mem = device.memory_info().ok()?;
    let name = device.name().ok()?;
    Some((mem.free, name, "NVML free VRAM"))
}

#[cfg(not(target_os = "macos"))]
fn rocm_probe() -> Option<(u64, String, &'static str)> {
    let out = std::process::Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--showproductname", "--json"])
        .output()
        .ok()?;
    let (free, name) = parse_rocm_smi(&String::from_utf8_lossy(&out.stdout))?;
    Some((free, name, "ROCm free VRAM (rocm-smi)"))
}

/// Pull free VRAM and a device name out of rocm-smi's JSON. Key names shift
/// between ROCm versions, so match on substrings of the first card's keys.
#[allow(dead_code)] // only wired into probe() off-macOS; unit-tested everywhere
fn parse_rocm_smi(json: &str) -> Option<(u64, String)> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let cards = v.as_object()?;
    let (_, card) = cards.iter().find(|(k, _)| k.starts_with("card"))?;
    let card = card.as_object()?;
    let grab = |needle: &str| -> Option<u64> {
        card.iter()
            .find(|(k, _)| k.contains(needle))
            .and_then(|(_, v)| v.as_str())
            .and_then(|s| s.trim().parse().ok())
    };
    let total = grab("VRAM Total Memory")?;
    let used = grab("VRAM Total Used Memory").unwrap_or(0);
    // Preference order matters: "Card model" is a bare hex id, not a name.
    let name = ["Card series", "Card SKU", "GPU"]
        .iter()
        .find_map(|needle| {
            card.iter().find(|(k, _)| k.contains(needle)).and_then(|(_, v)| v.as_str())
        })
        .unwrap_or("AMD GPU")
        .to_string();
    Some((total.saturating_sub(used), name))
}

#[cfg(test)]
mod tests {
    use super::parse_rocm_smi;

    #[test]
    fn parses_rocm_smi_json() {
        let json = r#"{
            "card0": {
                "VRAM Total Memory (B)": "17163091968",
                "VRAM Total Used Memory (B)": "1063091968",
                "Card series": "Radeon RX 7900 XTX",
                "Card model": "0x744c"
            }
        }"#;
        let (free, name) = parse_rocm_smi(json).unwrap();
        assert_eq!(free, 17163091968 - 1063091968);
        assert_eq!(name, "Radeon RX 7900 XTX");
    }

    #[test]
    fn rocm_parse_rejects_garbage() {
        assert!(parse_rocm_smi("not json").is_none());
        assert!(parse_rocm_smi("{}").is_none());
        assert!(parse_rocm_smi(r#"{"card0": {"unrelated": "1"}}"#).is_none());
    }
}

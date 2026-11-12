//! GPU-first with CPU fallback backend selection (Plan 5, task 5).
//!
//! The Wav2Lip + ExpNet models run on Burn, which can target either the GPU
//! (`burn-wgpu`, DX12/Vulkan on Windows) or CPU (`burn-flex`). This module
//! picks the backend automatically:
//!
//! - an explicit [`BackendPreference`] can force GPU or CPU,
//! - otherwise GPU is used when an adapter is available, CPU otherwise.
//!
//! Detection is done via [`Device::enumerate`], which is non-panicking and
//! returns an empty list when no adapter exists for the compiled backends.

use anyhow::{Context, Result};
use burn::prelude::*;

/// User preference for the execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Use the GPU when available, else CPU (default).
    #[default]
    Auto,
    /// Force the GPU (burn-wgpu); errors if no adapter is available.
    Gpu,
    /// Force the CPU (burn-flex).
    Cpu,
}

impl BackendPreference {
    /// Parses from a string (`"auto"`, `"gpu"`, `"cpu"`) — case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "gpu" | "cuda" | "vulkan" | "wgpu" => Some(Self::Gpu),
            "cpu" | "flex" | "ndarray" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// The selected backend + its Burn [`Device`].
#[derive(Debug, Clone)]
pub struct Backend {
    /// The device, on the GPU (burn-wgpu) or CPU (burn-flex).
    device: Device,
    /// Whether this is a GPU device.
    gpu: bool,
}

impl Backend {
    /// Selects a backend device according to `pref`.
    ///
    /// - [`BackendPreference::Gpu`]: returns an error when no GPU adapter is
    ///   available (rather than silently falling back).
    /// - [`BackendPreference::Cpu`]: always the flex CPU device.
    /// - [`BackendPreference::Auto`]: GPU when available, CPU otherwise.
    ///
    /// When only one backend is compiled in, the other is unavailable and the
    /// preference degrades gracefully.
    pub fn select(pref: BackendPreference) -> Result<Self> {
        match pref {
            BackendPreference::Cpu => Ok(Self::cpu()),
            BackendPreference::Gpu => {
                let device = Self::first_gpu()
                    .with_context(|| "GPU backend requested but no wgpu adapter is available")?;
                Ok(Self { device, gpu: true })
            }
            BackendPreference::Auto => {
                if let Some(device) = Self::first_gpu() {
                    Ok(Self { device, gpu: true })
                } else {
                    Ok(Self::cpu())
                }
            }
        }
    }

    /// Convenience: [`Backend::select`] with [`BackendPreference::Auto`].
    pub fn auto() -> Result<Self> {
        Self::select(BackendPreference::Auto)
    }

    /// The Burn [`Device`] to create tensors / load models on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether the selected device is a GPU (burn-wgpu).
    pub fn is_gpu(&self) -> bool {
        self.gpu
    }

    /// Human-readable backend name for logs.
    pub fn name(&self) -> &'static str {
        if self.gpu {
            "burn-wgpu (GPU)"
        } else {
            "burn-flex (CPU)"
        }
    }

    /// The first available GPU adapter, if any. Non-panicking.
    ///
    /// `DeviceType::Wgpu` uses wgpu's auto graphics-API selection (DX12 /
    /// Vulkan / Metal / WebGPU), which is what the plan's "GPU via burn-wgpu"
    /// means. The `gpu` feature compiles in `burn/wgpu`, so only this type is
    /// enumerated (the pinned Vulkan/Metal/WebGPU types need their own Burn
    /// features, which this crate does not enable).
    #[cfg(feature = "gpu")]
    fn first_gpu() -> Option<Device> {
        use burn::tensor::DeviceType;
        Device::enumerate(DeviceType::Wgpu)
            .into_vec()
            .into_iter()
            .next()
    }

    /// No GPU backend compiled in — nothing to enumerate.
    #[cfg(not(feature = "gpu"))]
    fn first_gpu() -> Option<Device> {
        None
    }

    /// CPU fallback (burn-flex).
    fn cpu() -> Self {
        Self {
            device: Device::flex(),
            gpu: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T5: forced CPU returns a flex device.
    #[test]
    fn force_cpu() {
        let b = Backend::select(BackendPreference::Cpu).unwrap();
        assert!(!b.is_gpu());
        assert_eq!(b.name(), "burn-flex (CPU)");
    }

    /// T5: parsing is case-insensitive and covers the aliases.
    #[test]
    fn preference_parse() {
        assert_eq!(
            BackendPreference::parse("auto"),
            Some(BackendPreference::Auto)
        );
        assert_eq!(BackendPreference::parse("GPU"), Some(BackendPreference::Gpu));
        assert_eq!(BackendPreference::parse("Cpu"), Some(BackendPreference::Cpu));
        assert_eq!(
            BackendPreference::parse("vulkan"),
            Some(BackendPreference::Gpu)
        );
        assert_eq!(BackendPreference::parse("flex"), Some(BackendPreference::Cpu));
        assert_eq!(BackendPreference::parse("wat"), None);
    }
}

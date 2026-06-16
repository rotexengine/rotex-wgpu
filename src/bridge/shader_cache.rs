use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use rotex_types::shader::ShaderPackage;

use crate::error::{Error, ErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ShaderSourceKind {
    Wgsl,
    Spirv,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ShaderCacheKey {
    kind: ShaderSourceKind,
    hash: u64,
}

pub(super) fn get_or_create_shader_module_from_package(
    bridge: &mut super::WgpuBridge,
    label: &'static str,
    package: &ShaderPackage,
) -> Result<wgpu::ShaderModule, Error> {
    if let Some(source) = package.wgsl_source() {
        return get_or_create_shader_module(bridge, label, Some(source), &[]);
    }
    if let Some(spirv) = package.spirv_bytes() {
        return get_or_create_shader_module(bridge, label, None, spirv);
    }
    Err(Error::recoverable(ErrorKind::InvalidDescriptor(
        "shader_variant_missing",
    )))
}

pub(super) fn get_or_create_shader_module(
    bridge: &mut super::WgpuBridge,
    label: &'static str,
    wgsl: Option<&str>,
    spirv: &[u8],
) -> Result<wgpu::ShaderModule, Error> {
    let (kind, hash, wgsl_source) = if let Some(source) = wgsl {
        (
            ShaderSourceKind::Wgsl,
            hash_bytes(source.as_bytes()),
            Some(source),
        )
    } else {
        if spirv.is_empty() {
            return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                "shader_bytes_missing",
            )));
        }
        if spirv.len() % 4 != 0 {
            return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                "shader_bytes_not_word_aligned",
            )));
        }
        (ShaderSourceKind::Spirv, hash_bytes(spirv), None)
    };

    let key = ShaderCacheKey { kind, hash };
    if let Some(module) = bridge.shader_module_cache.get(&key) {
        return Ok(module.clone());
    }

    let module = match kind {
        ShaderSourceKind::Wgsl => {
            let Some(source) = wgsl_source else {
                return Err(Error::fatal(ErrorKind::Unsupported(
                    "shader_wgsl_source_missing",
                )));
            };
            bridge.device.raw.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        }
        ShaderSourceKind::Spirv => bridge.device.raw.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::util::make_spirv(spirv),
            },
        ),
    };
    bridge.shader_module_cache.insert(key, module.clone());
    Ok(module)
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

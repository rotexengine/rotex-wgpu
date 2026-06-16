use rotex_types::shader::{ShaderPackage, ShaderPayload};

use crate::error::{Error, ErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ShaderCacheKey(u64);

pub(super) fn get_or_create_shader_module(
    bridge: &mut super::WgpuBridge,
    label: &'static str,
    package: &ShaderPackage,
) -> Result<wgpu::ShaderModule, Error> {
    let key = ShaderCacheKey(package.payload_hash());

    if let Some(module) = bridge.shader_module_cache.get(&key) {
        return Ok(module.clone());
    }

    let payload = package
        .variants
        .wgsl
        .as_ref()
        .or(package.variants.spirv.as_ref())
        .ok_or_else(|| {
            Error::recoverable(ErrorKind::InvalidDescriptor("shader_variant_missing"))
        })?;

    let module = match payload {
        ShaderPayload::Wgsl(source) => {
            bridge.device.raw.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.clone().into()),
            })
        }
        ShaderPayload::SpirV(bytes) => {
            if bytes.len() % 4 != 0 {
                return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                    "shader_bytes_not_word_aligned",
                )));
            }
            bridge.device.raw.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::util::make_spirv(bytes),
            })
        }
        _ => {
            return Err(Error::fatal(ErrorKind::Unsupported(
                "unsupported_shader_payload_for_wgpu",
            )));
        }
    };

    bridge.shader_module_cache.insert(key, module.clone());
    Ok(module)
}

use std::collections::HashMap;

use crate::backend::wgpu::WgpuInstance;
use crate::error::Error;
use rotex_types::{DeviceDescriptor, InstanceDescriptor};

use super::WgpuBridge;
use super::types::ResourceStorage;

pub(super) async fn create_bridge(
    instance_descriptor: InstanceDescriptor,
    device_descriptor: DeviceDescriptor,
) -> Result<WgpuBridge, Error> {
    let instance = WgpuInstance::new(&instance_descriptor).await?;
    let device = instance.request_device(&device_descriptor).await?;
    let texture_bind_group_layout = create_texture_bind_group_layout(&device.raw);
    let texture_sampler = create_texture_sampler(&device.raw);
    let fallback_texture_bind_group =
        create_fallback_texture_bind_group(&device, &texture_bind_group_layout, &texture_sampler);

    Ok(WgpuBridge {
        instance,
        device,
        texture_bind_group_layout,
        texture_sampler,
        fallback_texture_bind_group,
        surface: None,
        swapchain: None,
        resources: ResourceStorage::default(),
        next_mesh_id: 1,
        next_material_id: 1,
        next_texture_id: 1,
        pipeline_cache: HashMap::new(),
        depth_target: None,
    })
}

fn create_texture_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rotex-wgpu-texture-bind-group-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_texture_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("rotex-wgpu-texture-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

fn create_fallback_texture_bind_group(
    device: &crate::backend::wgpu::WgpuDevice,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    texture_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let texture = device.raw.create_texture(&wgpu::TextureDescriptor {
        label: Some("rotex-wgpu-fallback-white-texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    device.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255, 255, 255, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    device.raw.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rotex-wgpu-fallback-white-bind-group"),
        layout: texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(texture_sampler),
            },
        ],
    })
}

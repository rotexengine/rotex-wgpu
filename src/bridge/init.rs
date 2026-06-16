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

    Ok(WgpuBridge {
        instance,
        device,
        surface: None,
        swapchain: None,
        resources: ResourceStorage::default(),
        depth_target: None,
        rhi_pipeline_cache: HashMap::new(),
        shader_module_cache: HashMap::new(),
        compute_bind_groups: HashMap::new(),
        recording_encoder: None,
        surface_texture: None,
        swapchain_color_view: None,
        swapchain_format: None,
        current_frame_index: 0,
        active_pass: None,
        render_bundle: None,
        pending_acquire_skips: 0,
        pre_present_hook: None,
    })
}

use crate::error::{Error, ErrorKind};
use rotex_types::resource::ComputePipelineDescriptor;

use super::WgpuBridge;
use super::bindings;
use super::shader_cache;
use super::types::WgpuComputePipelineResource;

pub(super) fn create_compute_pipeline(
    bridge: &mut WgpuBridge,
    descriptor: &ComputePipelineDescriptor,
) -> Result<WgpuComputePipelineResource, Error> {
    let shader_package = &descriptor.shader;
    if shader_package.entry_point.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "compute_shader_entry_missing",
        )));
    }

    let bind_group_layouts =
        bindings::build_set_layouts_from_abstract_layout(&bridge.device.raw, &shader_package.layout);
    let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> =
        bind_group_layouts.iter().map(Some).collect();
    let pipeline_layout =
        bridge
            .device
            .raw
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rotex-wgpu-compute-pipeline-layout"),
                bind_group_layouts: &layout_refs,
                immediate_size: 0,
            });
    let shader = shader_cache::get_or_create_shader_module_from_package(
        bridge,
        "rotex-wgpu-compute-shader",
        shader_package,
    )?;
    let pipeline = bridge
        .device
        .raw
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rotex-wgpu-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(shader_package.entry_point.as_str()),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    Ok(WgpuComputePipelineResource {
        _descriptor: descriptor.clone(),
        pipeline,
        bind_group_layouts,
    })
}

pub(super) fn record_compute_dispatch(
    bridge: &mut super::WgpuBridge,
    encoder: &mut wgpu::CommandEncoder,
    pass: &rotex_types::ComputePassDescriptor,
) -> Result<(), Error> {
    let pipeline_resource = bridge
        .resources
        .compute_pipelines
        .get(&pass.pipeline)
        .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("compute_pipeline")))?;

    let mut intents_by_set = std::collections::BTreeMap::new();
    for intent in &pass.buffer_intents {
        intents_by_set
            .entry(intent.set)
            .or_insert_with(Vec::new)
            .push(intent);
    }

    let cache_key = (
        pass.pipeline,
        pass.buffer_intents
            .first()
            .map(|intent| intent.buffer)
            .ok_or_else(|| {
                Error::recoverable(ErrorKind::InvalidDescriptor(
                    "compute_pass_missing_buffer_intent",
                ))
            })?,
    );
    if !bridge.compute_bind_groups.contains_key(&cache_key) {
        let mut bind_groups = Vec::new();
        for (set_index, intents) in &intents_by_set {
            let layout = pipeline_resource
                .bind_group_layouts
                .get(*set_index as usize)
                .ok_or_else(|| {
                    Error::recoverable(ErrorKind::InvalidDescriptor(
                        "compute_bind_group_layout_missing",
                    ))
                })?;
            let mut entries = Vec::with_capacity(intents.len());
            for intent in intents.iter() {
                let buffer = bridge
                    .resources
                    .buffers
                    .get(&intent.buffer)
                    .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("buffer")))?;
                entries.push(wgpu::BindGroupEntry {
                    binding: intent.binding,
                    resource: wgpu::BindingResource::Buffer(compute_buffer_binding(
                        buffer, intent,
                    )),
                });
            }
            let bind_group = bridge
                .device
                .raw
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rotex-wgpu-compute-bind-group"),
                    layout,
                    entries: &entries,
                });
            bind_groups.push((*set_index, bind_group));
        }
        bridge.compute_bind_groups.insert(cache_key, bind_groups);
    }

    let bind_groups = bridge
        .compute_bind_groups
        .get(&cache_key)
        .ok_or_else(|| {
            Error::fatal(ErrorKind::Unsupported("compute_bind_group_cache_missing"))
        })?;

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rotex-wgpu-compute-pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline_resource.pipeline);
    for (set_index, bind_group) in bind_groups.iter() {
        compute_pass.set_bind_group(*set_index, bind_group, &[]);
    }
    compute_pass.dispatch_workgroups(
        pass.workgroup_count[0],
        pass.workgroup_count[1],
        pass.workgroup_count[2],
    );
    Ok(())
}

fn compute_buffer_binding<'a>(
    buffer: &'a super::types::WgpuBufferResource,
    intent: &rotex_types::BufferUsageIntent,
) -> wgpu::BufferBinding<'a> {
    if intent.size == 0 {
        wgpu::BufferBinding {
            buffer: &buffer.buffer,
            offset: intent.offset,
            size: None,
        }
    } else {
        wgpu::BufferBinding {
            buffer: &buffer.buffer,
            offset: intent.offset,
            size: std::num::NonZeroU64::new(intent.size),
        }
    }
}

use crate::error::{Error, ErrorKind};
use rotex_types::MaterialId;

use super::types::{MaterialPipelineKey, WgpuVertexLayout};
use super::{WgpuBridge, surface_not_attached_error};

pub(super) fn pipeline_for_draw<'a>(
    bridge: &'a mut WgpuBridge,
    material_id: MaterialId,
    vertex_layout_id: u64,
    vertex_layout: &WgpuVertexLayout,
    pass_uses_depth: bool,
) -> Result<&'a wgpu::RenderPipeline, Error> {
    let material = bridge
        .resources
        .materials
        .get(&material_id)
        .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("material")))?;
    let key = MaterialPipelineKey {
        material_id,
        vertex_layout_id,
        depth_enabled: pass_uses_depth && material.enable_depth,
    };

    if !bridge.pipeline_cache.contains_key(&key) {
        let pipeline = build_pipeline(bridge, material_id, vertex_layout, key.depth_enabled)?;
        bridge.pipeline_cache.insert(key, pipeline);
    }

    bridge.pipeline_cache.get(&key).ok_or_else(|| {
        Error::fatal(ErrorKind::PipelineCreationFailed(
            "missing_pipeline_after_insert",
        ))
    })
}

pub(super) fn invalidate_all(bridge: &mut WgpuBridge) {
    bridge.pipeline_cache.clear();
}

fn build_pipeline(
    bridge: &WgpuBridge,
    material_id: MaterialId,
    vertex_layout: &WgpuVertexLayout,
    depth_enabled: bool,
) -> Result<wgpu::RenderPipeline, Error> {
    let swapchain = bridge
        .swapchain
        .as_ref()
        .ok_or_else(surface_not_attached_error)?;
    let material = bridge
        .resources
        .materials
        .get(&material_id)
        .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("material")))?;
    let wgpu_cull_mode = match material.cull_mode {
        rotex_types::CullMode::None => None,
        rotex_types::CullMode::Front => Some(wgpu::Face::Front),
        rotex_types::CullMode::Back => Some(wgpu::Face::Back),
    };

    let vertex_spv = material.shaders.vertex.spirv_bytes().ok_or_else(|| {
        Error::recoverable(ErrorKind::InvalidDescriptor("vertex_shader_no_spirv"))
    })?;
    let fragment_spv = material.shaders.fragment.spirv_bytes().ok_or_else(|| {
        Error::recoverable(ErrorKind::InvalidDescriptor("fragment_shader_no_spirv"))
    })?;
    let vertex_shader = create_shader_module_from_spirv(
        &bridge.device.raw,
        "rotex-wgpu-vertex-shader",
        vertex_spv,
    )?;
    let fragment_shader = create_shader_module_from_spirv(
        &bridge.device.raw,
        "rotex-wgpu-fragment-shader",
        fragment_spv,
    )?;
    let layout = bridge
        .device
        .raw
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rotex-wgpu-pipeline-layout"),
            bind_group_layouts: &[Some(&bridge.texture_bind_group_layout)],
            immediate_size: 0,
        });

    let vertex_layout = vertex_layout.as_wgpu();

    let pipeline = bridge
        .device
        .raw
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rotex-wgpu-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some(material.shaders.vertex.entry_point.as_str()),
                buffers: &[vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu_cull_mode,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: if depth_enabled {
                Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                })
            } else {
                None
            },
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some(material.shaders.fragment.entry_point.as_str()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

    Ok(pipeline)
}

fn create_shader_module_from_spirv(
    device: &wgpu::Device,
    label: &'static str,
    spirv_bytes: &[u8],
) -> Result<wgpu::ShaderModule, Error> {
    if spirv_bytes.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "material_shader_bytes_missing",
        )));
    }
    if spirv_bytes.len() % 4 != 0 {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "material_shader_bytes_not_word_aligned",
        )));
    }

    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::util::make_spirv(spirv_bytes),
    }))
}

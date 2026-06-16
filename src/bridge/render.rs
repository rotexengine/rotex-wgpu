use crate::error::{Error, ErrorKind};
use rotex_types::{FrameDescriptor, SceneDescriptor};

use super::WgpuBridge;
use super::pipeline_cache;
use super::surface;
use super::types::DepthTarget;

struct DrawItem {
    pipeline: wgpu::RenderPipeline,
    texture_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    index_count: u32,
}

pub(super) fn render(
    bridge: &mut WgpuBridge,
    scene: &SceneDescriptor,
    frame: &FrameDescriptor,
) -> Result<(), Error> {
    if frame.passes.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "frame_has_no_passes",
        )));
    }

    let Some(surface_texture) = surface::acquire_surface_texture(bridge)? else {
        return Ok(());
    };
    let color_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let surface_format = bridge
        .swapchain
        .as_ref()
        .ok_or_else(super::surface_not_attached_error)?
        .config
        .format;

    let mut encoder = bridge
        .device
        .raw
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rotex-wgpu-encoder"),
        });

    for pass_outer in &frame.passes {
        let rotex_types::FramePass::Graphics(pass) = pass_outer else { continue; };
        let draw_list: Vec<usize> = if pass.instance_indices.is_empty() {
            (0..scene.instances.len()).collect()
        } else {
            pass.instance_indices
                .iter()
                .copied()
                .filter(|index| *index < scene.instances.len())
                .collect()
        };

        let pass_uses_depth = pass.uses_depth_attachment()
            || draw_list.iter().any(|index| {
                let instance = scene.instances[*index];
                bridge
                    .resources
                    .materials
                    .get(&instance.material)
                    .map(|material| material.enable_depth)
                    .unwrap_or(false)
            });
        let draw_items = collect_draw_items(bridge, scene, &draw_list, pass_uses_depth)?;

        let depth_attachment = if pass_uses_depth {
            let depth_view = ensure_depth_target(bridge)?;
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(pass.clear_depth),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rotex-wgpu-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color_for_surface(
                        surface_format,
                        pass.clear_color,
                    )),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        for item in draw_items {
            render_pass.set_pipeline(&item.pipeline);
            render_pass.set_bind_group(0, &item.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, item.vertex_buffer.slice(..));
            render_pass.set_index_buffer(item.index_buffer.slice(..), item.index_format);
            render_pass.draw_indexed(0..item.index_count, 0, 0..1);
        }
    }

    bridge.device.queue.submit(Some(encoder.finish()));
    surface_texture.present();
    Ok(())
}

fn ensure_depth_target(bridge: &mut WgpuBridge) -> Result<&wgpu::TextureView, Error> {
    let swapchain = bridge
        .swapchain
        .as_ref()
        .ok_or_else(super::surface_not_attached_error)?;
    let current_size = (
        swapchain.config.width.max(1),
        swapchain.config.height.max(1),
    );
    let needs_recreate = bridge
        .depth_target
        .as_ref()
        .map(|target| target.size != current_size)
        .unwrap_or(true);

    if needs_recreate {
        let texture = bridge.device.raw.create_texture(&wgpu::TextureDescriptor {
            label: Some("rotex-wgpu-depth"),
            size: wgpu::Extent3d {
                width: current_size.0,
                height: current_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        bridge.depth_target = Some(DepthTarget {
            _texture: texture,
            view,
            size: current_size,
        });
    }

    Ok(&bridge
        .depth_target
        .as_ref()
        .expect("depth target created")
        .view)
}

fn collect_draw_items(
    bridge: &mut WgpuBridge,
    scene: &SceneDescriptor,
    draw_list: &[usize],
    pass_uses_depth: bool,
) -> Result<Vec<DrawItem>, Error> {
    let mut draw_items = Vec::with_capacity(draw_list.len());
    for index in draw_list.iter().copied() {
        let instance = scene.instances[index];
        let texture_id = bridge
            .resources
            .materials
            .get(&instance.material)
            .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("material")))?
            .texture;
        let (
            vertex_layout_id,
            vertex_layout,
            vertex_buffer,
            index_buffer,
            index_format,
            index_count,
        ) = bridge
            .resources
            .meshes
            .get(&instance.mesh)
            .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("mesh")))?
            .to_draw_inputs();
        let pipeline = pipeline_cache::pipeline_for_draw(
            bridge,
            instance.material,
            vertex_layout_id,
            &vertex_layout,
            pass_uses_depth,
        )?
        .clone();
        let texture_bind_group = match texture_id {
            Some(texture_id) => bridge
                .resources
                .textures
                .get(&texture_id)
                .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("texture")))?
                .bind_group
                .clone(),
            None => bridge.fallback_texture_bind_group.clone(),
        };
        draw_items.push(DrawItem {
            pipeline,
            texture_bind_group,
            vertex_buffer,
            index_buffer,
            index_format,
            index_count,
        });
    }
    Ok(draw_items)
}

fn to_wgpu_color(clear: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: clear[0] as f64,
        g: clear[1] as f64,
        b: clear[2] as f64,
        a: clear[3] as f64,
    }
}

fn clear_color_for_surface(format: wgpu::TextureFormat, clear: [f32; 4]) -> wgpu::Color {
    let converted = if format.is_srgb() {
        clear
    } else {
        [
            linear_to_srgb(clear[0]),
            linear_to_srgb(clear[1]),
            linear_to_srgb(clear[2]),
            clear[3],
        ]
    };
    to_wgpu_color(converted)
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

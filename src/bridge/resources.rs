use crate::error::{Error, ErrorKind};
use rotex_types::{
    CreatedResources, MaterialDescriptor, MaterialId, MeshDescriptor, MeshId, ResourceBatchCreate,
    ResourceBatchUpdate, ResourceCreateDescriptor, ResourceHandle, ResourceUpdateDescriptor,
    TextureDescriptor, TextureFormat, TextureId, VertexBufferLayout, VertexFormat,
    VertexStreamData,
};
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hasher};

use super::WgpuBridge;
use super::types::{
    WgpuMeshResource, WgpuTextureResource, WgpuVertexLayout, bytes_per_pixel, index_format_size,
    map_index_format, map_texture_format, vertex_format_size, wgpu_vertex_attribute,
};

pub(super) fn create_resources(
    bridge: &mut WgpuBridge,
    descriptor: ResourceBatchCreate,
) -> Result<CreatedResources, Error> {
    let mut handles = Vec::with_capacity(descriptor.resources.len());
    for resource in descriptor.resources {
        match resource {
            ResourceCreateDescriptor::Mesh(mesh) => {
                let id = MeshId(bridge.next_mesh_id);
                bridge.next_mesh_id += 1;
                bridge
                    .resources
                    .meshes
                    .insert(id, create_wgpu_mesh(&bridge.device.raw, &mesh)?);
                handles.push(ResourceHandle::Mesh(id));
            }
            ResourceCreateDescriptor::Material(material) => {
                validate_material_descriptor(&material)?;
                let id = MaterialId(bridge.next_material_id);
                bridge.next_material_id += 1;
                bridge.resources.materials.insert(id, material);
                handles.push(ResourceHandle::Material(id));
            }
            ResourceCreateDescriptor::Texture(texture) => {
                let id = TextureId(bridge.next_texture_id);
                bridge.next_texture_id += 1;
                let gpu_texture = create_wgpu_texture(
                    &bridge.device,
                    texture,
                    &bridge.texture_bind_group_layout,
                    &bridge.texture_sampler,
                )?;
                bridge.resources.textures.insert(id, gpu_texture);
                handles.push(ResourceHandle::Texture(id));
            }
            _ => {}
        }
    }
    Ok(CreatedResources { handles })
}

pub(super) fn update_resources(
    bridge: &mut WgpuBridge,
    descriptor: ResourceBatchUpdate,
) -> Result<(), Error> {
    for update in descriptor.updates {
        match update {
            ResourceUpdateDescriptor::Mesh {
                id,
                vertex_streams,
                index_data,
                index_format,
                index_count,
            } => {
                let mesh = MeshDescriptor {
                    vertex_streams,
                    index_data,
                    index_format,
                    index_count,
                };
                let gpu_mesh = create_wgpu_mesh(&bridge.device.raw, &mesh)?;
                bridge.resources.meshes.insert(id, gpu_mesh);
            }
            ResourceUpdateDescriptor::Material {
                id,
                enable_depth,
                texture,
            } => {
                let material =
                    bridge.resources.materials.get_mut(&id).ok_or_else(|| {
                        Error::recoverable(ErrorKind::ResourceNotFound("material"))
                    })?;
                if let Some(depth) = enable_depth {
                    material.enable_depth = depth;
                }
                if let Some(texture_update) = texture {
                    if let Some(texture_id) = texture_update {
                        if !bridge.resources.textures.contains_key(&texture_id) {
                            return Err(Error::recoverable(ErrorKind::ResourceNotFound("texture")));
                        }
                    }
                    material.texture = texture_update;
                }
                bridge.pipeline_cache.retain(|key, _| key.material_id != id);
            }
            ResourceUpdateDescriptor::Texture { id, data } => {
                let texture =
                    bridge.resources.textures.get(&id).ok_or_else(|| {
                        Error::recoverable(ErrorKind::ResourceNotFound("texture"))
                    })?;
                write_texture_data(
                    &bridge.device,
                    texture.format,
                    texture.size.0,
                    texture.size.1,
                    &texture.texture,
                    &data,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn create_wgpu_mesh(
    device: &wgpu::Device,
    mesh: &MeshDescriptor,
) -> Result<WgpuMeshResource, Error> {
    if mesh.index_count == 0 {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "mesh_missing_indices",
        )));
    }
    if mesh.index_data.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "mesh_missing_index_data",
        )));
    }

    let (vertex_data, vertex_layout) = match &mesh.vertex_streams[0].data {
        VertexStreamData::Static(data) => {
            if data.is_empty() {
                return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                    "mesh_missing_vertex_data",
                )));
            }
            (data.as_slice(), &mesh.vertex_streams[0].layout)
        }
        VertexStreamData::External(_) => (&[][..], &mesh.vertex_streams[0].layout),
    };

    let index_stride = index_format_size(mesh.index_format);
    let expected_index_len = (mesh.index_count as usize)
        .checked_mul(index_stride)
        .ok_or_else(|| Error::recoverable(ErrorKind::InvalidDescriptor("index_size_overflow")))?;
    if mesh.index_data.len() != expected_index_len {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "mesh_index_data_size_mismatch",
        )));
    }

    let vertex_layout = translate_vertex_layout(vertex_layout)?;

    let vertex_buffer = create_buffer(
        device,
        wgpu::BufferUsages::VERTEX,
        vertex_data,
        "rotex-wgpu-vertex-buffer",
    );
    let index_buffer = create_buffer(
        device,
        wgpu::BufferUsages::INDEX,
        mesh.index_data.as_slice(),
        "rotex-wgpu-index-buffer",
    );

    Ok(WgpuMeshResource {
        vertex_buffer,
        index_buffer,
        index_format: map_index_format(mesh.index_format),
        index_count: mesh.index_count,
        vertex_layout_id: hash_vertex_layout(&mesh.vertex_streams[0].layout),
        vertex_layout,
    })
}

fn create_buffer<T: Copy>(
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    data: &[T],
    label: &str,
) -> wgpu::Buffer {
    let size = std::mem::size_of_val(data) as u64;
    let aligned_size = align_to_u64(size.max(1), wgpu::COPY_BUFFER_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: aligned_size,
        usage,
        mapped_at_creation: true,
    });
    if size > 0 {
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size as usize) };
        let mut staged = vec![0_u8; aligned_size as usize];
        staged[..bytes.len()].copy_from_slice(bytes);
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.copy_from_slice(staged.as_slice());
    }
    buffer.unmap();
    buffer
}

fn create_wgpu_texture(
    device: &crate::backend::wgpu::WgpuDevice,
    texture: TextureDescriptor,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    texture_sampler: &wgpu::Sampler,
) -> Result<WgpuTextureResource, Error> {
    let format = map_texture_format(texture.format);
    let width = texture.width.max(1);
    let height = texture.height.max(1);
    let raw_texture = device.raw.create_texture(&wgpu::TextureDescriptor {
        label: Some("rotex-wgpu-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    write_texture_data(device, format, width, height, &raw_texture, &texture.data)?;

    let view = raw_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.raw.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rotex-wgpu-texture-bind-group"),
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
    });
    Ok(WgpuTextureResource {
        texture: raw_texture,
        view,
        bind_group,
        format,
        size: (width, height),
    })
}

struct TextureUploadPlan {
    height: u32,
    bytes_per_row: u32,
    rows_per_image: u32,
    expected_len: usize,
    extent: wgpu::Extent3d,
}

fn write_texture_data(
    device: &crate::backend::wgpu::WgpuDevice,
    texture_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    texture: &wgpu::Texture,
    data: &[u8],
) -> Result<(), Error> {
    let upload = validate_texture_upload(texture_format, width, height, data)?;
    let src = &data[..upload.expected_len];
    if should_use_staging_upload(upload.bytes_per_row) {
        write_texture_data_with_staging(device, texture, src, &upload);
    } else {
        write_texture_data_with_queue(device, texture, src, &upload);
    }
    Ok(())
}

fn write_texture_data_with_queue(
    device: &crate::backend::wgpu::WgpuDevice,
    texture: &wgpu::Texture,
    data: &[u8],
    upload: &TextureUploadPlan,
) {
    device.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(upload.bytes_per_row),
            rows_per_image: Some(upload.rows_per_image),
        },
        upload.extent,
    );
}

fn write_texture_data_with_staging(
    device: &crate::backend::wgpu::WgpuDevice,
    texture: &wgpu::Texture,
    data: &[u8],
    upload: &TextureUploadPlan,
) {
    // Mirrors Vulkan-style uploads: host-visible staging buffer -> copy command into device-local texture.
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = align_to(upload.bytes_per_row, alignment);
    let mut staged = vec![0_u8; padded_bytes_per_row as usize * upload.height as usize];
    for row in 0..upload.height as usize {
        let src_offset = row * upload.bytes_per_row as usize;
        let dst_offset = row * padded_bytes_per_row as usize;
        staged[dst_offset..dst_offset + upload.bytes_per_row as usize]
            .copy_from_slice(&data[src_offset..src_offset + upload.bytes_per_row as usize]);
    }

    let staging_buffer = device.raw.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rotex-wgpu-texture-staging"),
        size: staged.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    staging_buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(staged.as_slice());
    staging_buffer.unmap();

    let mut encoder = device
        .raw
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rotex-wgpu-texture-upload-encoder"),
        });
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(upload.rows_per_image),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        upload.extent,
    );
    device.queue.submit(Some(encoder.finish()));
}

fn validate_material_descriptor(material: &MaterialDescriptor) -> Result<(), Error> {
    let vert_spv = material.shaders.vertex.spirv_bytes().ok_or_else(|| {
        Error::recoverable(ErrorKind::InvalidDescriptor("material_shader_bytes_missing"))
    })?;
    let frag_spv = material.shaders.fragment.spirv_bytes().ok_or_else(|| {
        Error::recoverable(ErrorKind::InvalidDescriptor("material_shader_bytes_missing"))
    })?;
    if material.shaders.vertex.entry_point.is_empty() || material.shaders.fragment.entry_point.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "material_shader_entry_missing",
        )));
    }
    if vert_spv.len() % 4 != 0 || frag_spv.len() % 4 != 0 {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "material_shader_bytes_not_word_aligned",
        )));
    }
    Ok(())
}

fn translate_vertex_layout(layout: &VertexBufferLayout) -> Result<WgpuVertexLayout, Error> {
    if layout.array_stride == 0 {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "vertex_layout_zero_stride",
        )));
    }
    if layout.attributes.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "vertex_layout_missing_attributes",
        )));
    }

    let mut attribute_locations = HashSet::with_capacity(layout.attributes.len());
    let mut translated = Vec::with_capacity(layout.attributes.len());
    for attribute in layout.attributes.iter().cloned() {
        let size = vertex_format_size(attribute.format);
        let end = attribute.offset.checked_add(size).ok_or_else(|| {
            Error::recoverable(ErrorKind::InvalidDescriptor("vertex_attribute_overflow"))
        })?;
        if end > layout.array_stride {
            return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                "vertex_attribute_out_of_bounds",
            )));
        }
        if !attribute_locations.insert(attribute.location) {
            return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                "vertex_attribute_duplicate_location",
            )));
        }
        translated.push(wgpu_vertex_attribute(attribute));
    }

    Ok(WgpuVertexLayout {
        array_stride: layout.array_stride,
        attributes: translated,
    })
}

fn hash_vertex_layout(layout: &VertexBufferLayout) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_u64(layout.array_stride);
    for attribute in &layout.attributes {
        hasher.write_u32(attribute.location);
        hasher.write_u64(attribute.offset);
        hasher.write_u8(vertex_format_tag(attribute.format));
    }
    hasher.finish()
}

fn vertex_format_tag(format: VertexFormat) -> u8 {
    match format {
        VertexFormat::Float32 => 0,
        VertexFormat::Float32x2 => 1,
        VertexFormat::Float32x3 => 2,
        VertexFormat::Float32x4 => 3,
        VertexFormat::Uint32 => 4,
    }
}

fn validate_texture_upload(
    texture_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<TextureUploadPlan, Error> {
    let mapped_format = match texture_format {
        wgpu::TextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        _ => {
            return Err(Error::recoverable(ErrorKind::Unsupported(
                "texture_format_not_supported",
            )));
        }
    };
    let width = width.max(1);
    let height = height.max(1);
    let bpp = bytes_per_pixel(mapped_format) as u64;
    let expected = (width as u64)
        .checked_mul(height as u64)
        .and_then(|value| value.checked_mul(bpp))
        .ok_or_else(|| Error::recoverable(ErrorKind::InvalidDescriptor("texture_size_overflow")))?;
    let expected_len = usize::try_from(expected)
        .map_err(|_| Error::recoverable(ErrorKind::InvalidDescriptor("texture_size_overflow")))?;
    if data.len() < expected_len {
        return Err(Error::recoverable(ErrorKind::TextureUploadFailed(
            "texture_data_too_small",
        )));
    }

    Ok(TextureUploadPlan {
        height,
        bytes_per_row: (width as u64 * bpp) as u32,
        rows_per_image: height,
        expected_len,
        extent: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    })
}

fn should_use_staging_upload(bytes_per_row: u32) -> bool {
    bytes_per_row.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

fn align_to(value: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

fn align_to_u64(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

use super::WgpuDevice;
use crate::error::{Error, ErrorKind};

pub struct WgpuSurface {
    raw: wgpu::Surface<'static>,
}

pub struct WgpuSwapchain {
    pub config: wgpu::SurfaceConfiguration,
}

impl WgpuSurface {
    pub fn new(raw: wgpu::Surface<'static>) -> Self {
        Self { raw }
    }

    pub fn create_swapchain(
        &self,
        device: &WgpuDevice,
        width: u32,
        height: u32,
    ) -> Result<WgpuSwapchain, Error> {
        let caps = self.raw.get_capabilities(&device.adapter);
        let preferred_formats = [
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        let format = preferred_formats
            .iter()
            .copied()
            .find(|candidate| caps.formats.contains(candidate))
            .or_else(|| caps.formats.iter().copied().find(wgpu::TextureFormat::is_srgb))
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| Error::fatal(ErrorKind::NoCompatibleDevice))?;
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            caps.present_modes
                .first()
                .copied()
                .ok_or_else(|| Error::fatal(ErrorKind::NoCompatibleDevice))?
        };
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes
                .first()
                .copied()
                .ok_or_else(|| Error::fatal(ErrorKind::NoCompatibleDevice))?
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        self.raw.configure(&device.raw, &config);
        Ok(WgpuSwapchain { config })
    }

    pub fn raw(&self) -> &wgpu::Surface<'static> {
        &self.raw
    }
}

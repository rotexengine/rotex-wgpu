#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTimingSnapshot {
    pub total_us: f64,
    pub surface_acquire_us: f64,
    pub record_commands_us: f64,
    pub transition_buffer_us: f64,
    pub dispatch_compute_us: f64,
    pub draw_graphics_us: f64,
    pub gfx_prepare_us: f64,
    pub gfx_render_pass_us: f64,
    pub queue_submit_us: f64,
    pub surface_present_us: f64,
    pub surface_acquire_skips: u32,
}

pub type FrameProfilerHook = Box<dyn Fn(FrameTimingSnapshot) + Send + 'static>;

pub fn add_elapsed(slot: &mut f64, start: Instant) {
    *slot += elapsed_us(start);
}

pub fn elapsed_us(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000_000.0
}

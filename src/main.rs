mod gpu_runtime;
mod tween;

use gpu_runtime::GpuRuntime;
use tween::*;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use std::sync::Arc;
use std::time::Instant;

const N: usize = 200;

struct Scene {
    compute_pipeline: i64,
    compute_bg: i64,
    render_pipeline: i64,
    render_bg: i64,
    items_buf: i64,
    params_buf: i64,
}

struct App {
    gpu: Option<GpuRuntime>,
    scene: Option<Scene>,
    window: Option<Arc<Window>>,
    particles: Vec<Particle>,
    shape: usize,
    last_time: Instant,
    needs_redraw: bool,
}

// ── Shape generators ──

fn shape_circle(i: usize, n: usize, cx: f64, cy: f64, r: f64) -> (f64, f64) {
    let angle = i as f64 / n as f64 * std::f64::consts::TAU;
    (cx + angle.cos() * r, cy + angle.sin() * r)
}

fn shape_square(i: usize, n: usize, cx: f64, cy: f64, size: f64) -> (f64, f64) {
    let half = size / 2.0;
    let perim = size * 4.0;
    let t = i as f64 / n as f64 * perim;
    if t < size { (cx - half + t, cy - half) }
    else if t < size * 2.0 { (cx + half, cy - half + (t - size)) }
    else if t < size * 3.0 { (cx + half - (t - size * 2.0), cy + half) }
    else { (cx - half, cy + half - (t - size * 3.0)) }
}

fn shape_star(i: usize, n: usize, cx: f64, cy: f64, r: f64) -> (f64, f64) {
    let points = 5;
    let seg = i / (n / points);
    let seg_t = (i % (n / points)) as f64 / (n / points) as f64;
    let a1 = seg as f64 / points as f64 * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
    let a2 = (seg as f64 + 0.5) / points as f64 * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
    let a3 = (seg + 1) as f64 / points as f64 * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
    let inner = r * 0.4;
    if seg_t < 0.5 {
        let t = seg_t * 2.0;
        let (x1, y1) = (cx + a1.cos() * r, cy + a1.sin() * r);
        let (x2, y2) = (cx + a2.cos() * inner, cy + a2.sin() * inner);
        (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
    } else {
        let t = (seg_t - 0.5) * 2.0;
        let (x1, y1) = (cx + a2.cos() * inner, cy + a2.sin() * inner);
        let (x2, y2) = (cx + a3.cos() * r, cy + a3.sin() * r);
        (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
    }
}

fn shape_spiral(i: usize, n: usize, cx: f64, cy: f64, r: f64) -> (f64, f64) {
    let t = i as f64 / n as f64;
    let angle = t * std::f64::consts::TAU * 3.0;
    let dist = t * r;
    (cx + angle.cos() * dist, cy + angle.sin() * dist)
}

fn shape_random(i: usize, _n: usize, cx: f64, cy: f64, r: f64) -> (f64, f64) {
    let a = ((i * 7919 + 104729) % 10000) as f64 / 10000.0;
    let b = ((i * 6271 + 37813) % 10000) as f64 / 10000.0;
    (cx + (a - 0.5) * r * 2.0, cy + (b - 0.5) * r * 2.0)
}

const SHAPE_FNS: &[fn(usize, usize, f64, f64, f64) -> (f64, f64)] = &[
    shape_circle, shape_square, shape_star, shape_spiral, shape_random,
];
const SHAPE_NAMES: &[&str] = &["circle", "square", "star", "spiral", "random"];

impl App {
    fn init_particles(&mut self, w: f64, h: f64) {
        let cx = w / 2.0;
        let cy = h / 2.0;
        let r = w.min(h) * 0.35;
        self.particles.clear();
        for i in 0..N {
            let (px, py) = shape_circle(i, N, cx, cy, r);
            let hue = i as f64 / N as f64 * 360.0;
            let delay = stagger_from_center(i, N, 3.0);
            self.particles.push(Particle {
                x: Tween::new(px, px, 1000.0, delay, ease_out_elastic),
                y: Tween::new(py, py, 1000.0, delay, ease_out_elastic),
                scale: Tween::new(1.0, 1.0, 800.0, delay, ease_out_cubic),
                opacity: Tween::new(1.0, 1.0, 600.0, delay, ease_out_cubic),
                hue: Tween::new(hue, hue + 60.0, 3000.0, delay, ease_in_out_sine),
            });
        }
        self.shape = 0;
    }

    fn morph_to(&mut self, shape_idx: usize) {
        let gpu = self.gpu.as_ref().unwrap();
        let w = gpu.width() as f64;
        let h = gpu.height() as f64;
        let cx = w / 2.0;
        let cy = h / 2.0;
        let r = w.min(h) * 0.35;
        let shape_fn = SHAPE_FNS[shape_idx];

        for (i, p) in self.particles.iter_mut().enumerate() {
            let (tx, ty) = shape_fn(i, N, cx, cy, r);
            let delay = stagger_from_center(i, N, 3.0);
            p.x = Tween::new(p.x.value(), tx, 1200.0, delay, ease_out_elastic);
            p.y = Tween::new(p.y.value(), ty, 1200.0, delay, ease_out_elastic);
            p.scale = Tween::new(0.3, 1.0, 800.0, delay, ease_out_cubic);
            p.opacity = Tween::new(0.2, 1.0, 600.0, delay, ease_out_cubic);
        }
    }

    fn upload_items(&mut self) {
        let gpu = self.gpu.as_mut().unwrap();
        let scene = self.scene.as_ref().unwrap();
        let count = self.particles.len();

        gpu.begin_data();
        for p in &self.particles {
            let x = p.x.value();
            let y = p.y.value();
            let s = p.scale.value();
            let o = p.opacity.value();
            let hue = p.hue.value();
            let r_size = 3.0 + s * 3.0;

            // Position as rect centered on (x, y)
            gpu.push_f32(x - r_size);
            gpu.push_f32(y - r_size);
            gpu.push_f32(r_size * 2.0);
            gpu.push_f32(r_size * 2.0);

            // HSL → RGB (simplified)
            let (r, g, b) = hsl_to_rgb(hue / 360.0, 0.8, 0.65);
            gpu.push_f32(r);
            gpu.push_f32(g);
            gpu.push_f32(b);
            gpu.push_f32(o);

            // Meta: rounded, opacity, scrollable, count
            gpu.push_f32(r_size); // fully rounded = circle
            gpu.push_f32(1.0);
            gpu.push_f32(0.0);
            gpu.push_f32(count as f64);
        }
        for _ in count..256 {
            for _ in 0..12 { gpu.push_f32(0.0); }
        }
        gpu.flush_to_buffer(scene.items_buf);
    }
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("ceangal-anime native — click to morph")
            .with_inner_size(winit::dpi::LogicalSize::new(600, 600));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let mut gpu = GpuRuntime::new();
        pollster::block_on(gpu.init(&window));

        let w = gpu.width();
        let h = gpu.height();

        let shader_src = include_str!("shader.wgsl").to_string();
        let shader_idx = gpu.register_shader_source(shader_src);
        let shader = gpu.create_shader(shader_idx);

        let pixel_buf = gpu.create_buffer((w * h * 4) as i64, 0x0080);
        let params_buf = gpu.create_buffer(16, 0x0040 | 0x0008);
        gpu.begin_data();
        gpu.push_u32(w as i64);
        gpu.push_u32(h as i64);
        gpu.push_u32(0);
        gpu.push_u32(0);
        gpu.flush_to_buffer(params_buf);

        let items_buf = gpu.create_buffer(256 * 48, 0x0080 | 0x0008);

        let cp = gpu.create_compute_pipeline(shader);
        gpu.begin_bindings();
        gpu.add_buffer_binding(pixel_buf);
        gpu.add_buffer_binding(params_buf);
        gpu.add_buffer_binding(items_buf);
        let cbg = gpu.create_bind_group_for_compute(cp, 0);

        let rp = gpu.create_render_pipeline(shader, 0);
        gpu.begin_bindings();
        gpu.add_buffer_binding(pixel_buf);
        gpu.add_buffer_binding(params_buf);
        let rbg = gpu.create_bind_group_for_render(rp, 0);

        self.gpu = Some(gpu);
        self.scene = Some(Scene {
            compute_pipeline: cp, compute_bg: cbg,
            render_pipeline: rp, render_bg: rbg,
            items_buf, params_buf,
        });
        self.window = Some(window);

        self.init_particles(w as f64, h as f64);
        self.needs_redraw = true;
        self.last_time = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::MouseInput { state: ElementState::Pressed, .. } => {
                self.shape = (self.shape + 1) % SHAPE_FNS.len();
                self.morph_to(self.shape);
                self.needs_redraw = true;
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_time).as_secs_f64() * 1000.0;
                self.last_time = now;

                // Tick all particles
                let mut any_moving = false;
                for p in &mut self.particles {
                    p.tick(dt);
                    let active = p.x.elapsed - p.x.delay;
                    if active < p.x.duration { any_moving = true; }
                }

                self.upload_items();

                let gpu = self.gpu.as_ref().unwrap();
                let scene = self.scene.as_ref().unwrap();
                gpu.render_frame(
                    scene.compute_pipeline, scene.compute_bg,
                    scene.render_pipeline, scene.render_bg,
                    (gpu.width() + 15) / 16, (gpu.height() + 15) / 16,
                );

                if any_moving {
                    self.window.as_ref().unwrap().request_redraw();
                } else {
                    self.needs_redraw = false;
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.needs_redraw {
            if let Some(w) = &self.window { w.request_redraw(); }
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        gpu: None, scene: None, window: None,
        particles: Vec::new(), shape: 0,
        last_time: Instant::now(), needs_redraw: false,
    };
    event_loop.run_app(&mut app).unwrap();
}

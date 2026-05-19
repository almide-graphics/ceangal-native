mod gpu_runtime;

use gpu_runtime::GpuRuntime;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use std::sync::Arc;

struct Scene {
    compute_pipeline: i64,
    compute_bg: i64,
    render_pipeline: i64,
    render_bg: i64,
}

struct App {
    gpu: Option<GpuRuntime>,
    scene: Option<Scene>,
    window: Option<Arc<Window>>,
}

impl App {
    fn setup_scene(&mut self) {
        let gpu = self.gpu.as_mut().unwrap();
        let w = gpu.width();
        let h = gpu.height();

        // Register shader
        let shader_src = include_str!("shader.wgsl").to_string();
        let shader_idx = gpu.register_shader_source(shader_src);
        let shader = gpu.create_shader(shader_idx);

        // Buffers
        let pixel_buf = gpu.create_buffer((w * h * 4) as i64, 0x0080); // STORAGE

        let params_buf = gpu.create_buffer(16, 0x0040 | 0x0008); // UNIFORM | COPY_DST
        gpu.begin_data();
        gpu.push_u32(w as i64);
        gpu.push_u32(h as i64);
        gpu.push_u32(0);
        gpu.push_u32(0);
        gpu.flush_to_buffer(params_buf);

        // Items: multiple rects to test
        let items_buf = gpu.create_buffer(256 * 48, 0x0080 | 0x0008); // STORAGE | COPY_DST
        let item_count = 5i64;
        gpu.begin_data();

        let rects: &[(f32, f32, f32, f32, f32, f32, f32, f32, f32)] = &[
            // (x, y, w, h, r, g, b, a, rounded)
            (50.0, 50.0, 200.0, 60.0, 0.15, 0.18, 0.25, 0.9, 8.0),    // header
            (50.0, 120.0, 200.0, 40.0, 0.12, 0.14, 0.22, 0.8, 6.0),   // input field
            (50.0, 170.0, 200.0, 40.0, 0.95, 0.95, 0.95, 0.08, 12.0), // item 1
            (50.0, 220.0, 200.0, 40.0, 0.95, 0.95, 0.95, 0.08, 12.0), // item 2
            (50.0, 270.0, 200.0, 40.0, 0.25, 0.50, 0.90, 1.0, 8.0),   // button
        ];

        for (i, &(x, y, rw, rh, r, g, b, a, rounded)) in rects.iter().enumerate() {
            gpu.push_f32(x as f64); gpu.push_f32(y as f64);
            gpu.push_f32(rw as f64); gpu.push_f32(rh as f64);
            gpu.push_f32(r as f64); gpu.push_f32(g as f64);
            gpu.push_f32(b as f64); gpu.push_f32(a as f64);
            gpu.push_f32(rounded as f64); gpu.push_f32(1.0); // opacity
            gpu.push_f32(0.0); // scrollable
            gpu.push_f32(item_count as f64); // count
        }
        // Pad remaining
        for _ in (item_count as usize)..256 {
            for _ in 0..12 { gpu.push_f32(0.0); }
        }
        gpu.flush_to_buffer(items_buf);

        // Compute pipeline + bind group
        let compute_pipeline = gpu.create_compute_pipeline(shader);
        gpu.begin_bindings();
        gpu.add_buffer_binding(pixel_buf);
        gpu.add_buffer_binding(params_buf);
        gpu.add_buffer_binding(items_buf);
        let compute_bg = gpu.create_bind_group_for_compute(compute_pipeline, 0);

        // Render pipeline + bind group
        let render_pipeline = gpu.create_render_pipeline(shader, 0);
        gpu.begin_bindings();
        gpu.add_buffer_binding(pixel_buf);
        gpu.add_buffer_binding(params_buf);
        let render_bg = gpu.create_bind_group_for_render(render_pipeline, 0);

        self.scene = Some(Scene {
            compute_pipeline,
            compute_bg,
            render_pipeline,
            render_bg,
        });
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_title("ceangal native");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let mut gpu = GpuRuntime::new();
        pollster::block_on(gpu.init(&window));
        self.gpu = Some(gpu);
        self.window = Some(window);
        self.setup_scene();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let (Some(gpu), Some(scene)) = (&self.gpu, &self.scene) {
                    let w = gpu.width();
                    let h = gpu.height();
                    gpu.render_frame(
                        scene.compute_pipeline,
                        scene.compute_bg,
                        scene.render_pipeline,
                        scene.render_bg,
                        (w + 15) / 16,
                        (h + 15) / 16,
                    );
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Don't spin — only redraw on events. Static content doesn't need continuous render.
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { gpu: None, scene: None, window: None };
    event_loop.run_app(&mut app).unwrap();
}

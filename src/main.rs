use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use std::sync::Arc;

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    pixel_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    items_buf: wgpu::Buffer,
    compute_bg: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl GpuState {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ).await.unwrap();

        let config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).unwrap();
        surface.configure(&device, &config);

        let w = config.width;
        let h = config.height;

        // Shader — minimal: compute writes pixels, fragment reads them
        let shader_src = include_str!("shader.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Buffers
        let pixel_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixels"),
            size: (w * h * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let params_data = [w, h, 0u32, 0]; // width, height, pad, pad
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Items: 1 rect (3 × vec4<f32> = 48 bytes per item, 256 items max)
        let mut items = vec![0.0f32; 256 * 12];
        // Item 0: x=100, y=100, w=300, h=200
        items[0] = 100.0; items[1] = 100.0; items[2] = 300.0; items[3] = 200.0;
        // bg color: r=0.3, g=0.5, b=0.9, a=1.0
        items[4] = 0.3; items[5] = 0.5; items[6] = 0.9; items[7] = 1.0;
        // meta: rounded=12, opacity=1, scrollable=0, count=1
        items[8] = 12.0; items[9] = 1.0; items[10] = 0.0; items[11] = 1.0;

        let items_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("items"),
            contents: bytemuck::cast_slice(&items),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute"),
            layout: None,
            module: &shader,
            entry_point: Some("fine"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pixel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: items_buf.as_entire_binding() },
            ],
        });

        // Render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_fullscreen"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &render_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pixel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });

        Self {
            surface, device, queue, config, compute_pipeline, render_pipeline,
            pixel_buf, params_buf, items_buf, compute_bg, render_bg,
            width: w, height: h,
        }
    }

    fn render(&self) {
        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Compute pass: items → pixel buffer
        {
            let mut cp = encoder.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.compute_pipeline);
            cp.set_bind_group(0, &self.compute_bg, &[]);
            cp.dispatch_workgroups(
                (self.width + 15) / 16,
                (self.height + 15) / 16,
                1,
            );
        }

        // Render pass: pixel buffer → screen
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.04, b: 0.07, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(0, &self.render_bg, &[]);
            rp.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

struct App {
    gpu: Option<GpuState>,
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_title("ceangal native");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.gpu = Some(pollster::block_on(GpuState::new(window.clone())));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = &self.gpu {
                    gpu.render();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { gpu: None, window: None };
    event_loop.run_app(&mut app).unwrap();
}

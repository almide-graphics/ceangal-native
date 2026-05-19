//! GPU runtime — wgpu implementation of snaidhm's 35 GPU functions.
//! Same handle-table pattern as the JS runtime.

use std::collections::HashMap;
use wgpu::util::DeviceExt;

// ── Handle table ──

#[derive(Default)]
pub struct GpuRuntime {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    format: Option<wgpu::TextureFormat>,

    // Handle table (0 = null)
    shaders: Vec<wgpu::ShaderModule>,
    buffers: Vec<wgpu::Buffer>,
    compute_pipelines: Vec<wgpu::ComputePipeline>,
    render_pipelines: Vec<wgpu::RenderPipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
    textures: Vec<wgpu::TextureView>,
    samplers: Vec<wgpu::Sampler>,
    encoders: Vec<Option<wgpu::CommandEncoder>>,
    // Passes stored as indices (active pass = last encoder)

    // Bind group builder state
    pending_bindings: Vec<BindingEntry>,

    // Data streaming state
    data_chunks: Vec<f32>,
    data_is_f32: Vec<bool>,

    // WGSL shader sources
    shader_sources: Vec<String>,
}

enum BindingEntry {
    Buffer(usize),    // index into buffers
    Texture(usize),   // index into textures
    Sampler(usize),   // index into samplers
}

// Pass wrapper — compute or render
enum ActivePass<'a> {
    Compute(wgpu::ComputePass<'a>),
    Render(wgpu::RenderPass<'a>),
}

impl GpuRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn init(&mut self, window: &std::sync::Arc<winit::window::Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        ).await.unwrap();

        let config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).unwrap();
        surface.configure(&device, &config);

        self.format = Some(config.format);
        self.surface_config = Some(config);
        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
    }

    fn device(&self) -> &wgpu::Device { self.device.as_ref().unwrap() }
    fn queue(&self) -> &wgpu::Queue { self.queue.as_ref().unwrap() }

    // ── 1. configure_canvas ──
    pub fn configure_canvas(&mut self, _format_handle: i64) -> i64 {
        // Already configured in init. Return dummy handle.
        0
    }

    // ── 2. get_preferred_format ──
    pub fn get_preferred_format(&self) -> i64 {
        0 // format stored internally
    }

    // ── 3. create_shader ──
    pub fn create_shader(&mut self, code_idx: i64) -> i64 {
        let source = &self.shader_sources[code_idx as usize];
        let module = self.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        self.shaders.push(module);
        (self.shaders.len() - 1) as i64
    }

    pub fn register_shader_source(&mut self, source: String) -> i64 {
        self.shader_sources.push(source);
        (self.shader_sources.len() - 1) as i64
    }

    // ── 4. create_buffer ──
    pub fn create_buffer(&mut self, size: i64, usage: i64) -> i64 {
        let buf = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::from_bits_truncate(usage as u32),
            mapped_at_creation: false,
        });
        self.buffers.push(buf);
        (self.buffers.len() - 1) as i64
    }

    // ── 5. write_buffer ──
    pub fn write_buffer(&self, buf_handle: i64, data: &[u8]) {
        self.queue().write_buffer(&self.buffers[buf_handle as usize], 0, data);
    }

    // ── 6. create_render_pipeline ──
    pub fn create_render_pipeline(&mut self, shader_handle: i64, format_handle: i64) -> i64 {
        let format = self.format.unwrap();
        let shader = &self.shaders[shader_handle as usize];
        let pipeline = self.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_fullscreen"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
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
        self.render_pipelines.push(pipeline);
        (self.render_pipelines.len() - 1) as i64
    }

    // ── 7. create_compute_pipeline ──
    pub fn create_compute_pipeline(&mut self, shader_handle: i64) -> i64 {
        let shader = &self.shaders[shader_handle as usize];
        let pipeline = self.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: shader,
            entry_point: Some("fine"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.compute_pipelines.push(pipeline);
        (self.compute_pipelines.len() - 1) as i64
    }

    // ── 8-10. Bind group builder ──
    pub fn begin_bindings(&mut self) {
        self.pending_bindings.clear();
    }

    pub fn add_buffer_binding(&mut self, buf_handle: i64) {
        self.pending_bindings.push(BindingEntry::Buffer(buf_handle as usize));
    }

    pub fn add_texture_binding(&mut self, tex_handle: i64) {
        self.pending_bindings.push(BindingEntry::Texture(tex_handle as usize));
    }

    pub fn add_sampler_binding(&mut self, samp_handle: i64) {
        self.pending_bindings.push(BindingEntry::Sampler(samp_handle as usize));
    }

    pub fn create_bind_group_for_compute(&mut self, pipeline_handle: i64, group_idx: u32) -> i64 {
        let pipeline = &self.compute_pipelines[pipeline_handle as usize];
        let layout = pipeline.get_bind_group_layout(group_idx);
        let entries: Vec<wgpu::BindGroupEntry> = self.pending_bindings.iter().enumerate().map(|(i, entry)| {
            wgpu::BindGroupEntry {
                binding: i as u32,
                resource: match entry {
                    BindingEntry::Buffer(idx) => self.buffers[*idx].as_entire_binding(),
                    BindingEntry::Texture(idx) => wgpu::BindingResource::TextureView(&self.textures[*idx]),
                    BindingEntry::Sampler(idx) => wgpu::BindingResource::Sampler(&self.samplers[*idx]),
                },
            }
        }).collect();
        let bg = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        self.bind_groups.push(bg);
        (self.bind_groups.len() - 1) as i64
    }

    pub fn create_bind_group_for_render(&mut self, pipeline_handle: i64, group_idx: u32) -> i64 {
        let pipeline = &self.render_pipelines[pipeline_handle as usize];
        let layout = pipeline.get_bind_group_layout(group_idx);
        let entries: Vec<wgpu::BindGroupEntry> = self.pending_bindings.iter().enumerate().map(|(i, entry)| {
            wgpu::BindGroupEntry {
                binding: i as u32,
                resource: match entry {
                    BindingEntry::Buffer(idx) => self.buffers[*idx].as_entire_binding(),
                    BindingEntry::Texture(idx) => wgpu::BindingResource::TextureView(&self.textures[*idx]),
                    BindingEntry::Sampler(idx) => wgpu::BindingResource::Sampler(&self.samplers[*idx]),
                },
            }
        }).collect();
        let bg = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        self.bind_groups.push(bg);
        (self.bind_groups.len() - 1) as i64
    }

    // ── 11-14. Data streaming ──
    pub fn begin_data(&mut self) {
        self.data_chunks.clear();
        self.data_is_f32.clear();
    }

    pub fn push_f32(&mut self, value: f64) {
        self.data_chunks.push(value as f32);
        self.data_is_f32.push(true);
    }

    pub fn push_u32(&mut self, value: i64) {
        self.data_chunks.push(f32::from_bits(value as u32));
        self.data_is_f32.push(false);
    }

    pub fn flush_to_buffer(&self, buf_handle: i64) {
        let buf = &self.buffers[buf_handle as usize];
        let bytes: Vec<u8> = self.data_chunks.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.queue().write_buffer(buf, 0, &bytes);
    }

    // ── 15-16. Partial buffer write ──
    pub fn write_f32_at(&self, buf_handle: i64, byte_offset: i64, value: f64) {
        let buf = &self.buffers[buf_handle as usize];
        self.queue().write_buffer(buf, byte_offset as u64, &(value as f32).to_ne_bytes());
    }

    pub fn write_u32_at(&self, buf_handle: i64, byte_offset: i64, value: i64) {
        let buf = &self.buffers[buf_handle as usize];
        self.queue().write_buffer(buf, byte_offset as u64, &(value as u32).to_ne_bytes());
    }

    // ── High-level render: compute + present ──
    pub fn render_frame(
        &self,
        compute_pipeline: i64,
        compute_bg: i64,
        render_pipeline: i64,
        render_bg: i64,
        tiles_x: u32,
        tiles_y: u32,
    ) {
        let surface = self.surface.as_ref().unwrap();
        let output = surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&Default::default());

        let mut encoder = self.device().create_command_encoder(&Default::default());

        // Compute pass
        {
            let mut cp = encoder.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.compute_pipelines[compute_pipeline as usize]);
            cp.set_bind_group(0, &self.bind_groups[compute_bg as usize], &[]);
            cp.dispatch_workgroups(tiles_x, tiles_y, 1);
        }

        // Render pass
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.04, b: 0.07, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_pipeline(&self.render_pipelines[render_pipeline as usize]);
            rp.set_bind_group(0, &self.bind_groups[render_bg as usize], &[]);
            rp.draw(0..6, 0..1);
        }

        self.queue().submit(std::iter::once(encoder.finish()));
        output.present();
    }

    pub fn width(&self) -> u32 { self.surface_config.as_ref().map(|c| c.width).unwrap_or(1) }
    pub fn height(&self) -> u32 { self.surface_config.as_ref().map(|c| c.height).unwrap_or(1) }
}

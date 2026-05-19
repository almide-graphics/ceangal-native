// ceangal-native minimal shader
// Compute: items → pixel buffer
// Fragment: pixel buffer → screen

struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(1) var<uniform>             params: Params;
@group(0) @binding(2) var<storage, read>       items: array<vec4<f32>>;

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
  let q = abs(p) - b + vec2<f32>(r, r);
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

fn pack_color(r: f32, g: f32, b: f32, a: f32) -> u32 {
  let ri = u32(clamp(r * 255.0, 0.0, 255.0));
  let gi = u32(clamp(g * 255.0, 0.0, 255.0));
  let bi = u32(clamp(b * 255.0, 0.0, 255.0));
  let ai = u32(clamp(a * 255.0, 0.0, 255.0));
  return ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}

@compute @workgroup_size(16, 16)
fn fine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= params.width || py >= params.height) { return; }

  let ri_count = u32(items[2].w);
  var color = vec3<f32>(0.0);
  var alpha = 0.0;

  for (var ri = 0u; ri < min(ri_count, 256u); ri++) {
    let pos = items[ri * 3u];
    let col = items[ri * 3u + 1u];
    let item_meta = items[ri * 3u + 2u];

    let ix = pos.x; let iy = pos.y;
    let iw = pos.z; let ih = pos.w;

    if iw < 1.0 || ih < 1.0 || col.w < 0.01 { continue; }

    let fpx = f32(px); let fpy = f32(py);
    if fpx < ix || fpx > ix + iw || fpy < iy || fpy > iy + ih { continue; }

    let corner_r = item_meta.x;
    let local = vec2<f32>(fpx - ix - iw * 0.5, fpy - iy - ih * 0.5);
    let half = vec2<f32>(iw * 0.5, ih * 0.5);
    let d = sd_rounded_box(local, half, corner_r);

    if d < 1.0 && col.w > 0.01 {
      let aa = 1.0 - smoothstep(-1.0, 0.5, d);
      let a = aa * col.w * item_meta.y;
      color = mix(color, col.xyz, a);
      alpha = alpha + a * (1.0 - alpha);
    }
  }

  pixels[py * params.width + px] = pack_color(color.x, color.y, color.z, alpha);
}

// Fullscreen quad
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VSOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),  vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  );
  var uvs = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
  out.uv = uvs[idx];
  return out;
}

@group(0) @binding(0) var<storage, read> render_pixels: array<u32>;
@group(0) @binding(1) var<uniform>       render_params: Params;

@fragment
fn fs_fullscreen(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let w = f32(render_params.width);
  let h = f32(render_params.height);
  let px_u = u32(uv.x * w);
  let py_u = u32(uv.y * h);

  let idx = py_u * render_params.width + px_u;
  let packed = render_pixels[idx];
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  let a = f32((packed >> 24u) & 0xFFu) / 255.0;

  // Dark background + items composited
  let bg = vec3<f32>(0.04, 0.04, 0.07);
  let final_color = mix(bg, vec3(r, g, b), a);

  return vec4<f32>(final_color, 1.0);
}

// Expression atlas warp shader (report section 3.6, plan task 6).
//
// For each output pixel, computes a displacement field by inverse-distance
// weighting over per-mode control points (mouth corners, brows, eyes, nose,
// chin), then inverse-maps to the source image and bilinear-samples it.
//
// The output texture is the base image height * 7 (one band per expression
// mode, stacked vertically). The variant index is derived from the output row.

struct Params {
    strength: f32,
    n_points: u32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;

fn sample_bilinear(uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src));
    let p = uv * dims;
    let p0 = floor(p);
    let f = p - p0;
    let max = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let x0 = min(i32(p0.x), max.x);
    let y0 = min(i32(p0.y), max.y);
    let x1 = min(x0 + 1, max.x);
    let y1 = min(y0 + 1, max.y);
    let c00 = textureLoad(src, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(src, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(src, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(src, vec2<i32>(x1, y1), 0);
    let c0 = mix(c00, c10, vec4<f32>(f.x));
    let c1 = mix(c01, c11, vec4<f32>(f.x));
    return mix(c0, c1, vec4<f32>(f.y));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src);
    let w = dims.x;
    let h = dims.y;
    let variant = gid.y / h;
    let y = gid.y % h;
    if (gid.x >= w || y >= h) {
        return;
    }
    let n = vec2<f32>(f32(w), f32(h));
    let pn = vec2<f32>(f32(gid.x), f32(y)) / n;

    var disp = vec2<f32>(0.0, 0.0);
    var wsum = 0.0;
    let base = variant * params.n_points;
    for (var i = 0u; i < params.n_points; i = i + 1u) {
        let cp = points[base + i];
        let d = distance(pn, cp.xy);
        let w = 1.0 / (d * d + 0.004);
        disp = disp + w * cp.zw;
        wsum = wsum + w;
    }
    disp = disp / wsum;

    let src_pn = clamp(pn - disp * params.strength, vec2<f32>(0.0), vec2<f32>(1.0));
    let color = sample_bilinear(src_pn);
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}

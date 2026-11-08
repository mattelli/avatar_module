// Avatar frame renderer (report section 3.6, plan task 7).
//
// Renders one avatar frame per invocation from the expression atlas
// (task 6) + per-frame parameters:
// - viseme weights (task 5) drive a jaw-drop stretch and a darkening
//   mouth ellipse that grows with mouth openness;
// - head bob / breathing (task 4) apply a subtle sway and scale;
// - fade_t cross-fades between two atlas bands (~150 ms at 25 fps,
//   computed by the pipeline glue in task 9).
//
// Output is size x size RGBA (32 or 64 px), one frame per dispatch.

struct U {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    head_bob: f32,
    breathing: f32,
    fade_t: f32,
    pad0: f32,
    variant_from: u32,
    variant_to: u32,
    pad1: vec2<f32>,
};

@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: U;

// Mouth openness per viseme, in VISEMES order:
// sil=0, AI=1, E=0.8, O=0.9, MBP=0, FV=0.15, L=0.3, TH=0.2, U=0.6, WQ=0.4, etc=0.25, rest=0
const OPEN0 = vec4<f32>(0.0, 1.0, 0.8, 0.9);
const OPEN1 = vec4<f32>(0.0, 0.15, 0.3, 0.2);
const OPEN2 = vec4<f32>(0.6, 0.4, 0.25, 0.0);

const MOUTH_Y = 0.72;
const MOUTH_HALF_W = 0.12;
const JAW = 0.16;
const BANDS = 7.0;

fn sample_band(band: u32, uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(atlas));
    let buv = vec2<f32>(uv.x, (f32(band) + uv.y) / BANDS);
    let p = buv * dims;
    let p0 = floor(p);
    let f = p - p0;
    let max = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let x0 = min(i32(p0.x), max.x);
    let y0 = min(i32(p0.y), max.y);
    let x1 = min(x0 + 1, max.x);
    let y1 = min(y0 + 1, max.y);
    let c00 = textureLoad(atlas, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(atlas, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(atlas, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(atlas, vec2<i32>(x1, y1), 0);
    let c0 = mix(c00, c10, vec4<f32>(f.x));
    let c1 = mix(c01, c11, vec4<f32>(f.x));
    return mix(c0, c1, vec4<f32>(f.y));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(atlas);
    let w = dims.x;
    let band_h = dims.y / 7u;
    if (gid.x >= w || gid.y >= band_h) {
        return;
    }
    let openness = clamp(dot(u.v0, OPEN0) + dot(u.v1, OPEN1) + dot(u.v2, OPEN2), 0.0, 1.0);

    var uv = vec2<f32>(f32(gid.x) / f32(w), f32(gid.y) / f32(band_h));

    // Breathing: gentle scale around the center.
    let scale = 1.0 + u.breathing * 0.008;
    uv = (uv - vec2<f32>(0.5)) / scale + vec2<f32>(0.5);

    // Head bob: subtle vertical sway plus a tiny horizontal component.
    uv.y = uv.y + u.head_bob * 0.02;
    uv.x = uv.x + u.head_bob * 0.004;

    // Viseme jaw drop: stretch the region below the mouth center.
    var sy = uv.y;
    let dx = abs(uv.x - 0.5);
    if (dx < MOUTH_HALF_W && uv.y > MOUTH_Y && uv.y < MOUTH_Y + JAW) {
        let d = uv.y - MOUTH_Y;
        sy = MOUTH_Y + d / (1.0 + openness * 0.35);
    }
    let suv = vec2<f32>(uv.x, sy);

    // Expression cross-fade between two atlas bands.
    let c_from = sample_band(u.variant_from, suv);
    let c_to = sample_band(u.variant_to, suv);
    var color = mix(c_from, c_to, u.fade_t);

    // Mouth opening: darken an ellipse that grows with openness.
    if (openness > 0.05) {
        let mw = 0.085 * (0.6 + openness);
        let mh = 0.008 + 0.045 * openness;
        let cx = (uv.x - 0.5) / mw;
        let cy = (uv.y - (MOUTH_Y + 0.012)) / mh;
        let e = cx * cx + cy * cy;
        if (e < 1.0) {
            let mouth_dark = vec4<f32>(0.16, 0.05, 0.05, 1.0);
            color = mix(color, mouth_dark, 0.75 * (1.0 - e));
        }
    }

    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}

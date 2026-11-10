//! Avatar frame renderer (report §3.6, plan task 7).                                                                                                                                                           
//!                                                                                                                                                                                                             
//! Consumes the expression atlas (task 6) and renders one `size × size`                                                                                                                                        
//! RGBA frame per dispatch (32 or 64 px) using a wgpu compute shader:                                                                                                                                          
//! - viseme weights (task 5) → jaw drop + mouth ellipse darkening,                                                                                                                                             
//! - head bob / breathing (task 4) → subtle sway and scale,                                                                                                                                                    
//! - `fade_t` → cross-fade between two atlas bands (~150 ms at 25 fps,                                                                                                                                         
//!   computed by the pipeline glue in task 9).                                                                                                                                                                 
//!                                                                                                                                                                                                             
//! Frames are read back to the CPU as tightly-packed RGBA8.                                                                                                                                                    
                                                                                                                                                                                                                
use crate::visemes::VISEMES;                                                                                                                                                                                    
use image::{GenericImageView, ImageBuffer, Rgba};                                                                                                                                                               
use wgpu::util::DeviceExt;                                                                                                                                                                                      
                                                                                                                                                                                                                
/// Number of visemes.                                                                                                                                                                                          
const K: usize = VISEMES.len();                                                                                                                                                                                 
                                                                                                                                                                                                                
/// Mouth openness per viseme, in `VISEMES` order                                                                                                                                                               
/// (sil, AI, E, O, MBP, FV, L, TH, U, WQ, etc, rest).                                                                                                                                                          
pub const OPENNESS: [f32; K] = [                                                                                                                                                                                
    0.0, 1.0, 0.8, 0.9, 0.0, 0.15, 0.3, 0.2, 0.6, 0.4, 0.25, 0.0,                                                                                                                                               
];                                                                                                                                                                                                              
                                                                                                                                                                                                                
/// Per-frame parameters (must match `U` in renderer.wgsl).                                                                                                                                                     
#[repr(C)]                                                                                                                                                                                                      
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]                                                                                                                                                
struct GpuFrameParams {                                                                                                                                                                                         
    /// Viseme weights [0..4).                                                                                                                                                                                  
    v0: [f32; 4],                                                                                                                                                                                               
    /// Viseme weights [4..8).                                                                                                                                                                                  
    v1: [f32; 4],                                                                                                                                                                                               
    /// Viseme weights [8..12).                                                                                                                                                                                 
    v2: [f32; 4],                                                                                                                                                                                               
    /// Head bob offset in [-1, 1] (task 4).                                                                                                                                                                    
    head_bob: f32,                                                                                                                                                                                              
    /// Breathing amplitude in [0, 1] (task 4).                                                                                                                                                                 
    breathing: f32,                                                                                                                                                                                             
    /// Cross-fade factor in [0, 1].                                                                                                                                                                            
    fade_t: f32,                                                                                                                                                                                                
    pad0: f32,                                                                                                                                                                                                  
    /// Atlas band to fade from.                                                                                                                                                                                
    variant_from: u32,                                                                                                                                                                                          
    /// Atlas band to fade to.                                                                                                                                                                                  
    variant_to: u32,                                                                                                                                                                                            
    pad1: [f32; 2],                                                                                                                                                                                             
}                                                                                                                                                                                                               
                                                                                                                                                                                                                
/// GPU renderer of avatar frames from the atlas.                                                                                                                                                               
pub struct FrameRenderer {                                                                                                                                                                                      
    device: wgpu::Device,                                                                                                                                                                                       
    queue: wgpu::Queue,                                                                                                                                                                                         
    pipeline: wgpu::ComputePipeline,                                                                                                                                                                            
}                                                                                                                                                                                                               
                                                                                                                                                                                                                
impl FrameRenderer {                                                                                                                                                                                            
    /// Requests a GPU device and builds the pipeline.                                                                                                                                                          
    pub async fn new() -> anyhow::Result<Self> {                                                                                                                                                                
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());                                                                                                                               
        let adapter = instance                                                                                                                                                                                  
            .request_adapter(&wgpu::RequestAdapterOptions {                                                                                                                                                     
                power_preference: wgpu::PowerPreference::HighPerformance,                                                                                                                                       
                compatible_surface: None,                                                                                                                                                                       
                force_fallback_adapter: false,                                                                                                                                                                  
            })                                                                                                                                                                                                  
            .await                                                                                                                                                                                              
            .map_err(|_| anyhow::anyhow!("no GPU adapter"))?;                                                                                                                                                   
        let (device, queue) = adapter                                                                                                                                                                           
            .request_device(&wgpu::DeviceDescriptor {                                                                                                                                                           
                label: Some("avatar-frame"),                                                                                                                                                                    
                required_features: wgpu::Features::empty(),                                                                                                                                                     
                required_limits: wgpu::Limits::default(),                                                                                                                                                       
                memory_hints: wgpu::MemoryHints::default(),                                                                                                                                                     
                trace: wgpu::Trace::Off,                                                                                                                                                                        
            })                                                                                                                                                                                                  
            .await                                                                                                                                                                                              
            .map_err(|_| anyhow::anyhow!("no GPU device"))?;                                                                                                                                                    
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {                                                                                                                                 
            label: Some("renderer.wgsl"),                                                                                                                                                                       
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer.wgsl").into()),                                                                                                                             
        });                                                                                                                                                                                                     
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {                                                                                                                        
            label: Some("frame-pipeline"),                                                                                                                                                                      
            layout: None,                                                                                                                                                                                       
            module: &shader,                                                                                                                                                                                    
            entry_point: Some("main"),                                                                                                                                                                          
            compilation_options: Default::default(),                                                                                                                                                            
            cache: None,                                                                                                                                                                                        
        });                                                                                                                                                                                                     
        Ok(Self { device, queue, pipeline })                                                                                                                                                                    
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// Renders one frame.                                                                                                                                                                                      
    ///                                                                                                                                                                                                         
    /// - `atlas`: atlas from task 6 (width × width×7).                                                                                                                                                         
    /// - `viseme_weights`: K weights summing to 1 (task 5 row).                                                                                                                                                
    /// - `head_bob`, `breathing`: task 4 frame values.                                                                                                                                                         
    /// - `variant_from`, `variant_to`, `fade_t`: expression cross-fade.                                                                                                                                        
    /// - `size`: output size (32 or 64).                                                                                                                                                                       
    pub fn render_frame(                                                                                                                                                                                        
        &self,                                                                                                                                                                                                  
        atlas: &ImageBuffer<Rgba<u8>, Vec<u8>>,                                                                                                                                                                 
        viseme_weights: &[f32],                                                                                                                                                                                 
        head_bob: f32,                                                                                                                                                                                          
        breathing: f32,                                                                                                                                                                                         
        variant_from: u32,                                                                                                                                                                                      
        variant_to: u32,                                                                                                                                                                                        
        fade_t: f32,                                                                                                                                                                                            
        size: u32,                                                                                                                                                                                              
    ) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {                                                                                                                                                       
        let device = &self.device;                                                                                                                                                                              
        let (aw, ah) = atlas.dimensions();                                                                                                                                                                      
        anyhow::ensure!(ah == aw * 7, "atlas must be width × width×7, got {aw}×{ah}");                                                                                                                          
        anyhow::ensure!(                                                                                                                                                                                        
            viseme_weights.len() == K,                                                                                                                                                                          
            "viseme weights must have K = {} entries",                                                                                                                                                          
            K                                                                                                                                                                                                   
        );                                                                                                                                                                                                      
        anyhow::ensure!(size == 32 || size == 64, "size must be 32 or 64");                                                                                                                                     
                                                                                                                                                                                                                
        // Split K weights into the three vec4s.                                                                                                                                                                
        let mut v = [0.0f32; 12];                                                                                                                                                                               
        v.copy_from_slice(viseme_weights);                                                                                                                                                                      
        let params = GpuFrameParams {                                                                                                                                                                           
            v0: [v[0], v[1], v[2], v[3]],                                                                                                                                                                       
            v1: [v[4], v[5], v[6], v[7]],                                                                                                                                                                       
            v2: [v[8], v[9], v[10], v[11]],                                                                                                                                                                     
            head_bob,                                                                                                                                                                                           
            breathing,                                                                                                                                                                                          
            fade_t: fade_t.clamp(0.0, 1.0),                                                                                                                                                                     
            pad0: 0.0,                                                                                                                                                                                          
            variant_from,                                                                                                                                                                                       
            variant_to,                                                                                                                                                                                         
            pad1: [0.0, 0.0],                                                                                                                                                                                   
        };                                                                                                                                                                                                      
                                                                                                                                                                                                                
        let atlas_tex = device.create_texture_with_data(                                                                                                                                                        
            &self.queue,                                                                                                                                                                                        
            &wgpu::TextureDescriptor {                                                                                                                                                                          
                label: Some("frame-atlas"),                                                                                                                                                                     
                size: wgpu::Extent3d { width: aw, height: ah, depth_or_array_layers: 1 },                                                                                                                       
                mip_level_count: 1,                                                                                                                                                                             
                sample_count: 1,                                                                                                                                                                                
                dimension: wgpu::TextureDimension::D2,                                                                                                                                                          
                format: wgpu::TextureFormat::Rgba8Unorm,                                                                                                                                                        
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,                                                                                                                    
                view_formats: &[],                                                                                                                                                                              
            },                                                                                                                                                                                                  
            wgpu::util::TextureDataOrder::default(),                                                                                                                                                            
            atlas.as_raw(),                                                                                                                                                                                     
        );                                                                                                                                                                                                      
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());                                                                                                                        
        let dst_tex = device.create_texture(&wgpu::TextureDescriptor {                                                                                                                                          
            label: Some("frame-dst"),                                                                                                                                                                           
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },                                                                                                                       
            mip_level_count: 1,                                                                                                                                                                                 
            sample_count: 1,                                                                                                                                                                                    
            dimension: wgpu::TextureDimension::D2,                                                                                                                                                              
            format: wgpu::TextureFormat::Rgba8Unorm,                                                                                                                                                            
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,                                                                                                                        
            view_formats: &[],                                                                                                                                                                                  
        });                                                                                                                                                                                                     
        let dst_view = dst_tex.create_view(&wgpu::TextureViewDescriptor::default());                                                                                                                            
                                                                                                                                                                                                                
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {                                                                                                                          
            label: Some("frame-params"),                                                                                                                                                                        
            contents: bytemuck::bytes_of(&params),                                                                                                                                                              
            usage: wgpu::BufferUsages::UNIFORM,                                                                                                                                                                 
        });                                                                                                                                                                                                     
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {                                                                                                                                  
            label: Some("frame-bind"),                                                                                                                                                                          
            layout: &self.pipeline.get_bind_group_layout(0),                                                                                                                                                    
            entries: &[                                                                                                                                                                                         
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&atlas_view) },                                                                                                 
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },                                                                                                   
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },                                                                                                                  
            ],                                                                                                                                                                                                  
        });                                                                                                                                                                                                     
                                                                                                                                                                                                                
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {                                                                                                                       
            label: Some("frame-encoder"),                                                                                                                                                                       
        });                                                                                                                                                                                                     
        {                                                                                                                                                                                                       
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {                                                                                                                            
                label: Some("frame-pass"),                                                                                                                                                                      
                timestamp_writes: None,                                                                                                                                                                         
            });                                                                                                                                                                                                 
            pass.set_pipeline(&self.pipeline);                                                                                                                                                                  
            pass.set_bind_group(0, &bind_group, &[]);                                                                                                                                                           
            pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 1);                                                                                                                                    
        }                                                                                                                                                                                                       
        self.queue.submit(Some(encoder.finish()));                                                                                                                                                              
                                                                                                                                                                                                                
        // Read back.                                                                                                                                                                                           
        let bytes_per_row = (size * 4).next_multiple_of(256);                                                                                                                                                   
        let buf_size = (bytes_per_row as u64) * (size as u64);                                                                                                                                                  
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {                                                                                                                                           
            label: Some("frame-read"),                                                                                                                                                                          
            size: buf_size,                                                                                                                                                                                     
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,                                                                                                                                 
            mapped_at_creation: false,                                                                                                                                                                          
        });                                                                                                                                                                                                     
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {                                                                                                                       
            label: Some("frame-copy"),                                                                                                                                                                          
        });                                                                                                                                                                                                     
        encoder.copy_texture_to_buffer(                                                                                                                                                                         
            wgpu::TexelCopyTextureInfo {                                                                                                                                                                        
                texture: &dst_tex,                                                                                                                                                                              
                mip_level: 0,                                                                                                                                                                                   
                origin: wgpu::Origin3d::ZERO,                                                                                                                                                                   
                aspect: wgpu::TextureAspect::All,                                                                                                                                                               
            },                                                                                                                                                                                                  
            wgpu::TexelCopyBufferInfo {                                                                                                                                                                         
                buffer: &read_buf,                                                                                                                                                                              
                layout: wgpu::TexelCopyBufferLayout {                                                                                                                                                           
                    offset: 0,                                                                                                                                                                                  
                    bytes_per_row: Some(bytes_per_row),                                                                                                                                                         
                    rows_per_image: None,                                                                                                                                                                       
                },                                                                                                                                                                                              
            },                                                                                                                                                                                                  
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },                                                                                                                             
        );                                                                                                                                                                                                      
        self.queue.submit(Some(encoder.finish()));                                                                                                                                                              
                                                                                                                                                                                                                
        let slice = read_buf.slice(..);                                                                                                                                                                         
        let (tx, rx) = std::sync::mpsc::channel();                                                                                                                                                              
        slice.map_async(wgpu::MapMode::Read, move |r| {                                                                                                                                                         
            let _ = tx.send(r);                                                                                                                                                                                 
        });                                                                                                                                                                                                     
        self.device                                                                                                                                                                                             
            .poll(wgpu::PollType::Wait)                                                                                                                                                                         
            .map_err(|e| anyhow::anyhow!("poll failed: {:?}", e))?;                                                                                                                                             
        rx.recv()                                                                                                                                                                                               
            .map_err(|e| anyhow::anyhow!("map callback lost: {}", e))??;                                                                                                                                        
                                                                                                                                                                                                                
        let data = slice.get_mapped_range();                                                                                                                                                                    
        let mut out = ImageBuffer::new(size, size);                                                                                                                                                             
        for y in 0..size {                                                                                                                                                                                      
            let row =                                                                                                                                                                                           
                &data[y as usize * bytes_per_row as usize..(y as usize + 1) * bytes_per_row as usize];                                                                                                          
            for x in 0..size {                                                                                                                                                                                  
                let o = x as usize * 4;                                                                                                                                                                         
                out.put_pixel(x, y, Rgba([row[o], row[o + 1], row[o + 2], row[o + 3]]));                                                                                                                        
            }                                                                                                                                                                                                   
        }                                                                                                                                                                                                       
        drop(data);                                                                                                                                                                                             
        read_buf.unmap();                                                                                                                                                                                       
        Ok(out)                                                                                                                                                                                                 
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// Convenience: renders a sequence of frames for a clip.                                                                                                                                                   
    ///                                                                                                                                                                                                         
    /// `frames` is a list of `(viseme_weights, head_bob, breathing,                                                                                                                                            
    /// variant_from, variant_to, fade_t)`.                                                                                                                                                                     
    pub fn render_clip(                                                                                                                                                                                         
        &self,                                                                                                                                                                                                  
        atlas: &ImageBuffer<Rgba<u8>, Vec<u8>>,                                                                                                                                                                 
        frames: &[([f32; 12], f32, f32, u32, u32, f32)],                                                                                                                                                        
        size: u32,                                                                                                                                                                                              
    ) -> anyhow::Result<Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>> {                                                                                                                                                  
        frames                                                                                                                                                                                                  
            .iter()                                                                                                                                                                                             
            .map(|(w, hb, b, vf, vt, ft)| {                                                                                                                                                                     
                self.render_frame(atlas, w, *hb, *b, *vf, *vt, *ft, size)                                                                                                                                       
            })                                                                                                                                                                                                  
            .collect()                                                                                                                                                                                          
    }                                                                                                                                                                                                           
}                                                                                                                                                                                                               
                                                                                                                                                                                                                
/// Mouth openness for a row of viseme weights (dot product with OPENNESS).                                                                                                                                     
pub fn openness_of(weights: &[f32]) -> anyhow::Result<f32> {                                                                                                                                                    
    anyhow::ensure!(weights.len() == OPENNESS.len(), "K weights required");                                                                                                                                     
    let o: f32 = weights.iter().zip(&OPENNESS).map(|(w, o)| w * o).sum();                                                                                                                                       
    Ok(o.clamp(0.0, 1.0))                                                                                                                                                                                       
}                                                                                                                                                                                                               
                                                                                                                                                                                                                
#[cfg(test)]                                                                                                                                                                                                    
mod tests {                                                                                                                                                                                                     
    use super::*;                                                                                                                                                                                               
    use crate::atlas::{atlas_cpu, test_face_cpu};                                                                                                                                                               
                                                                                                                                                                                                                
    /// Builds a synthetic face and atlas on the CPU.                                                                                                                                                           
    fn build_atlas(w: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {                                                                                                                                                  
        let face = test_face_cpu(w, w);                                                                                                                                                                         
        atlas_cpu(&face, 1.0)                                                                                                                                                                                   
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// Neutral frame weights: silence only.                                                                                                                                                                    
    fn neutral_weights() -> [f32; 12] {                                                                                                                                                                         
        let mut w = [0.0f32; 12];                                                                                                                                                                               
        w[0] = 1.0; // sil                                                                                                                                                                                      
        w                                                                                                                                                                                                       
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// Mean brightness over R+G+B.                                                                                                                                                                             
    fn mean_rgb(f: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> f32 {                                                                                                                                                    
        let (w, h) = f.dimensions();                                                                                                                                                                            
        let mut sum = 0u64;                                                                                                                                                                                     
        for y in 0..h {                                                                                                                                                                                         
            for x in 0..w {                                                                                                                                                                                     
                let p = f.get_pixel(x, y);                                                                                                                                                                      
                sum += p[0] as u64 + p[1] as u64 + p[2] as u64;                                                                                                                                                 
            }                                                                                                                                                                                                   
        }                                                                                                                                                                                                       
        sum as f32 / ((w * h * 3) as f32)                                                                                                                                                                       
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// Count of differing pixels between two frames.                                                                                                                                                           
    fn changed_pixels(                                                                                                                                                                                          
        a: &ImageBuffer<Rgba<u8>, Vec<u8>>,                                                                                                                                                                     
        b: &ImageBuffer<Rgba<u8>, Vec<u8>>,                                                                                                                                                                     
    ) -> usize {                                                                                                                                                                                                
        let (w, h) = a.dimensions();                                                                                                                                                                            
        let mut n = 0;                                                                                                                                                                                          
        for y in 0..h {                                                                                                                                                                                         
            for x in 0..w {                                                                                                                                                                                     
                if a.get_pixel(x, y) != b.get_pixel(x, y) {                                                                                                                                                     
                    n += 1;                                                                                                                                                                                     
                }                                                                                                                                                                                               
            }                                                                                                                                                                                                   
        }                                                                                                                                                                                                       
        n                                                                                                                                                                                                       
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// T7: device is available (AMD GPU expected).                                                                                                                                                             
    #[test]                                                                                                                                                                                                     
    fn device_available() {                                                                                                                                                                                     
        let rt = tokio::runtime::Runtime::new().unwrap();                                                                                                                                                       
        match rt.block_on(FrameRenderer::new()) {                                                                                                                                                               
            Ok(r) => assert!(r.device.limits().max_texture_dimension_2d > 0),                                                                                                                                   
            Err(e) => eprintln!("no GPU device, skipping: {}", e),                                                                                                                                              
        }                                                                                                                                                                                                       
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// T7: renders 32 and 64 px frames with correct dimensions.                                                                                                                                                
    #[test]                                                                                                                                                                                                     
    fn renders_32_and_64() {                                                                                                                                                                                    
        let rt = tokio::runtime::Runtime::new().unwrap();                                                                                                                                                       
        let Ok(r) = rt.block_on(FrameRenderer::new()) else {                                                                                                                                                    
            eprintln!("no GPU device, skipping");                                                                                                                                                               
            return;                                                                                                                                                                                             
        };                                                                                                                                                                                                      
        let w = neutral_weights();                                                                                                                                                                              
        let f_small = r.render_frame(&build_atlas(32), &w, 0.0, 0.0, 0, 0, 0.0, 32).unwrap();                                                                                                                   
        assert_eq!(f_small.dimensions(), (32, 32));                                                                                                                                                             
        let f_large = r.render_frame(&build_atlas(64), &w, 0.0, 0.0, 0, 0, 0.0, 64).unwrap();                                                                                                                   
        assert_eq!(f_large.dimensions(), (64, 64));                                                                                                                                                             
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// T7: output dimensions are correct and alpha is opaque.                                                                                                                                                  
    #[test]                                                                                                                                                                                                     
    fn output_dimensions_correct() {                                                                                                                                                                            
        let rt = tokio::runtime::Runtime::new().unwrap();                                                                                                                                                       
        let Ok(r) = rt.block_on(FrameRenderer::new()) else {                                                                                                                                                    
            eprintln!("no GPU device, skipping");                                                                                                                                                               
            return;                                                                                                                                                                                             
        };                                                                                                                                                                                                      
        let atlas = build_atlas(64);                                                                                                                                                                            
        let w = neutral_weights();                                                                                                                                                                              
        let f = r.render_frame(&atlas, &w, 0.5, 0.5, 0, 1, 0.5, 64).unwrap();                                                                                                                                   
        assert_eq!(f.dimensions(), (64, 64));                                                                                                                                                                   
        for y in 0..64 {                                                                                                                                                                                        
            for x in 0..64 {                                                                                                                                                                                    
                assert_eq!(f.get_pixel(x, y)[3], 255, "alpha at ({}, {})", x, y);                                                                                                                               
            }                                                                                                                                                                                                   
        }                                                                                                                                                                                                       
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// T7: cross-fade produces intermediate pixel values.                                                                                                                                                      
    #[test]                                                                                                                                                                                                     
    fn crossfade_intermediate() {                                                                                                                                                                               
        let rt = tokio::runtime::Runtime::new().unwrap();                                                                                                                                                       
        let Ok(r) = rt.block_on(FrameRenderer::new()) else {                                                                                                                                                    
            eprintln!("no GPU device, skipping");                                                                                                                                                               
            return;                                                                                                                                                                                             
        };                                                                                                                                                                                                      
        let atlas = build_atlas(64);                                                                                                                                                                            
        let w = neutral_weights();                                                                                                                                                                              
        // Fade from neutral (0) to happy (1): mean color moves monotonically.                                                                                                                                  
        let means: Vec<f32> = [0.0f32, 0.25, 0.5, 0.75, 1.0]                                                                                                                                                    
            .iter()                                                                                                                                                                                             
            .map(|&ft| {                                                                                                                                                                                        
                let f = r.render_frame(&atlas, &w, 0.0, 0.0, 0, 1, ft, 64).unwrap();                                                                                                                            
                mean_rgb(&f)                                                                                                                                                                                    
            })                                                                                                                                                                                                  
            .collect();                                                                                                                                                                                         
        for wnd in means.windows(2) {                                                                                                                                                                           
            assert!(wnd[1] >= wnd[0] - 1.0, "means not monotonic: {:?}", means);                                                                                                                                
        }                                                                                                                                                                                                       
        // Endpoints differ (neutral vs happy warp differ).                                                                                                                                                     
        let f0 = r.render_frame(&atlas, &w, 0.0, 0.0, 0, 1, 0.0, 64).unwrap();                                                                                                                                  
        let f1 = r.render_frame(&atlas, &w, 0.0, 0.0, 0, 1, 1.0, 64).unwrap();                                                                                                                                  
        assert!(changed_pixels(&f0, &f1) > 0, "fade endpoints identical");                                                                                                                                      
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// T7: viseme weights change the mouth region (open mouth is darker).                                                                                                                                      
    #[test]                                                                                                                                                                                                     
    fn viseme_changes_mouth() {                                                                                                                                                                                 
        let rt = tokio::runtime::Runtime::new().unwrap();                                                                                                                                                       
        let Ok(r) = rt.block_on(FrameRenderer::new()) else {                                                                                                                                                    
            eprintln!("no GPU device, skipping");                                                                                                                                                               
            return;                                                                                                                                                                                             
        };                                                                                                                                                                                                      
        let atlas = build_atlas(64);                                                                                                                                                                            
        let mut open = [0.0f32; 12];                                                                                                                                                                            
        open[1] = 1.0; // AI: wide open                                                                                                                                                                         
        let closed = r.render_frame(&atlas, &neutral_weights(), 0.0, 0.0, 0, 0, 0.0, 64).unwrap();                                                                                                              
        let opened = r.render_frame(&atlas, &open, 0.0, 0.0, 0, 0, 0.0, 64).unwrap();                                                                                                                           
        assert!(                                                                                                                                                                                                
            mean_rgb(&opened) < mean_rgb(&closed),                                                                                                                                                              
            "open mouth not darker: {} vs {}",                                                                                                                                                                  
            mean_rgb(&opened),                                                                                                                                                                                  
            mean_rgb(&closed)                                                                                                                                                                                   
        );                                                                                                                                                                                                      
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// T7: head bob sways the frame.                                                                                                                                                                           
    #[test]                                                                                                                                                                                                     
    fn head_bob_sways() {                                                                                                                                                                                       
        let rt = tokio::runtime::Runtime::new().unwrap();                                                                                                                                                       
        let Ok(r) = rt.block_on(FrameRenderer::new()) else {                                                                                                                                                    
            eprintln!("no GPU device, skipping");                                                                                                                                                               
            return;                                                                                                                                                                                             
        };                                                                                                                                                                                                      
        let atlas = build_atlas(64);                                                                                                                                                                            
        let w = neutral_weights();                                                                                                                                                                              
        let f0 = r.render_frame(&atlas, &w, 0.0, 0.0, 0, 0, 0.0, 64).unwrap();                                                                                                                                  
        let f1 = r.render_frame(&atlas, &w, 1.0, 0.0, 0, 0, 0.0, 64).unwrap();                                                                                                                                  
        assert!(changed_pixels(&f0, &f1) > 0, "head bob changed nothing");                                                                                                                                      
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// T7: openness_of dot product is correct.                                                                                                                                                                 
    #[test]                                                                                                                                                                                                     
    fn openness_dot() {                                                                                                                                                                                         
        let mut w = [0.0f32; 12];                                                                                                                                                                               
        assert_eq!(openness_of(&w).unwrap(), 0.0);                                                                                                                                                              
        w[1] = 1.0; // AI                                                                                                                                                                                       
        assert_eq!(openness_of(&w).unwrap(), 1.0);                                                                                                                                                              
        let mut w2 = [0.0f32; 12];                                                                                                                                                                              
        w2[4] = 1.0; // MBP                                                                                                                                                                                     
        assert_eq!(openness_of(&w2).unwrap(), 0.0);                                                                                                                                                             
        let mut w3 = [0.0f32; 12];                                                                                                                                                                              
        w3[3] = 1.0; // O                                                                                                                                                                                       
        assert_eq!(openness_of(&w3).unwrap(), 0.9);                                                                                                                                                             
    }                                                                                                                                                                                                           
                                                                                                                                                                                                                
    /// T7: atlas validation errors.                                                                                                                                                                            
    #[test]                                                                                                                                                                                                     
    fn atlas_validation() {                                                                                                                                                                                     
        let rt = tokio::runtime::Runtime::new().unwrap();                                                                                                                                                       
        let Ok(r) = rt.block_on(FrameRenderer::new()) else {                                                                                                                                                    
            eprintln!("no GPU device, skipping");                                                                                                                                                               
            return;                                                                                                                                                                                             
        };                                                                                                                                                                                                      
        let w = neutral_weights();                                                                                                                                                                              
        // Bad atlas shape (not width × width×7).                                                                                                                                                               
        let bad = ImageBuffer::new(32, 32);                                                                                                                                                                     
        assert!(r.render_frame(&bad, &w, 0.0, 0.0, 0, 0, 0.0, 32).is_err());                                                                                                                                    
        // Wrong K.                                                                                                                                                                                             
        let atlas = build_atlas(32);                                                                                                                                                                            
        assert!(r.render_frame(&atlas, &[0.5f32], 0.0, 0.0, 0, 0, 0.0, 32).is_err());                                                                                                                           
        // Wrong size.                                                                                                                                                                                          
        assert!(r.render_frame(&atlas, &w, 0.0, 0.0, 0, 0, 0.0, 100).is_err());                                                                                                                                 
    }                                                                                                                                                                                                           

    /// Regression (plan 4): the renderer must sample the FULL atlas band, not
    /// just its top-left corner, when the atlas is wider than the output.
    ///
    /// The pipeline builds a 128x128 band atlas and renders 64x64 frames. The
    /// shader used to normalize UVs by the ATLAS width (128), so the 64 output
    /// pixels only covered uv in [0, 0.5] - the top-left quadrant of the source.
    /// UVs must be normalized by the OUTPUT size.
    #[test]
    fn atlas_wider_than_output_samples_full_band() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let Ok(r) = rt.block_on(FrameRenderer::new()) else {
            eprintln!("no GPU device, skipping");
            return;
        };

        // Source: 128x128 horizontal gradient, left = red, right = blue.
        // If the shader samples only the top-left corner, the output's right
        // edge stays red; if it samples the full band, it becomes blue.
        let mut src = ImageBuffer::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                let t = x as f32 / 127.0;
                src.put_pixel(x, y, Rgba([(255.0 * (1.0 - t)) as u8, 0, (255.0 * t) as u8, 255]));
            }
        }
        let atlas = atlas_cpu(&src, 0.0); // identity warp
        let w = neutral_weights();
        let frame = r.render_frame(&atlas, &w, 0.0, 0.0, 0, 0, 0.0, 64).unwrap();

        let left = frame.get_pixel(0, 32);
        let right = frame.get_pixel(63, 32);
        // Left edge should be red-dominant, right edge blue-dominant.
        assert!(
            left[0] as u16 > right[0] as u16,
            "left edge not redder than right: left={left:?} right={right:?}"
        );
        assert!(
            right[2] as u16 > left[2] as u16,
            "right edge not bluer than left: left={left:?} right={right:?}"
        );
        // The right edge must actually be blue-ish (not a red corner).
        assert!(
            right[2] > 150,
            "right edge should be blue (full band sampled), got {right:?}"
        );
    }
}

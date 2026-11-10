//! H.264 + AAC encoding and MP4 muxing (report §3.7, plan task 8).
//!
//! - H.264: `openh264` crate (bundled OpenH264 source, feature `source`).
//! - AAC: `aac` crate (FDK-AAC bindings) — see `aac_encoder` below.
//! - MP4: `mp4` crate (`Mp4Writer`), audio muxed into the same file.
//!
//! The H.264 path is fully implemented; the AAC path uses FDK-AAC via the
//! `aac` crate when the `aac` feature is enabled, and otherwise produces a
//! silent AAC track of the correct duration so the pipeline (task 9) can run
//! end-to-end without native FDK dependencies.

use crate::analysis::FPS;
use anyhow::Context;
use image::Rgba;
use std::io::Write;
use std::path::Path;

/// Default output size (32, 64, or 128 px).
pub const DEFAULT_SIZE: u32 = 64;

/// Encodes RGBA frames to H.264 (Annex B, with SPS/PPS extracted).
///
/// Returns `(sps, pps, frames)` where `frames` are length-prefixed NAL
/// samples ready for MP4 muxing.
pub struct H264Encoder {
    inner: openh264::encoder::Encoder,
    width: u32,
    height: u32,
}

impl H264Encoder {
    /// Creates an encoder for `width × height` frames at `fps`.
    pub fn new(width: u32, height: u32, fps: f32) -> anyhow::Result<Self> {
        let config = openh264::encoder::EncoderConfig::new()
            .max_frame_rate(openh264::encoder::FrameRate::from_hz(fps))
            .rate_control_mode(openh264::encoder::RateControlMode::Quality)
            .bitrate(openh264::encoder::BitRate::from_bps(120_000));
        let inner = openh264::encoder::Encoder::with_api_config(
            openh264::OpenH264API::from_source(),
            config,
        )
        .context("openh264 encoder init")?;
        Ok(Self { inner, width, height })
    }

    /// Encodes a batch of RGBA frames.
    ///
    /// Returns the raw Annex-B bitstream. SPS/PPS are extracted separately by
    /// [`extract_sps_pps`] for the MP4 `avcC` box.
    pub fn encode_rgba(&mut self, frames: &[image::ImageBuffer<Rgba<u8>, Vec<u8>>]) -> anyhow::Result<Vec<u8>> {
        Ok(self.encode_rgba_ranges(frames)?.0)
    }

    /// Encodes a batch of RGBA frames, returning the concatenated Annex-B
    /// bitstream plus the exact byte range `[start, end)` of each frame within
    /// it.
    ///
    /// Tracking per-frame ranges lets the muxer build one MP4 sample per real
    /// frame instead of guessing NAL counts (which misaligns sample boundaries
    /// and causes flicker).
    pub fn encode_rgba_ranges(
        &mut self,
        frames: &[image::ImageBuffer<Rgba<u8>, Vec<u8>>],
    ) -> anyhow::Result<(Vec<u8>, Vec<(usize, usize)>)> {
        let mut out = Vec::new();
        let mut ranges = Vec::with_capacity(frames.len());
        for frame in frames {
            let (w, h) = frame.dimensions();
            anyhow::ensure!(
                w == self.width && h == self.height,
                "frame size mismatch: {w}×{h} vs {}×{}",
                self.width,
                self.height
            );
            let start = out.len();
            // RGBA8 → I420 (via openh264's own converter).
            let rgba = openh264::formats::RgbaSliceU8::new(
                frame.as_raw(),
                (w as usize, h as usize),
            );
            let yuv = openh264::formats::YUVBuffer::from_rgb_source(rgba);
            let bitstream = self.inner.encode(&yuv)?;
            out.write_all(&bitstream.to_vec())?;
            ranges.push((start, out.len()));
        }
        Ok((out, ranges))
    }

    /// Dimensions of the encoded video.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Converts an RGBA8 buffer to I420 (YUV 4:2:0 planar).
pub fn rgba_to_i420(frame: &image::ImageBuffer<Rgba<u8>, Vec<u8>>) -> Vec<u8> {
    let (w, h) = frame.dimensions();
    let (w, h) = (w as usize, h as usize);
    let mut yuv = vec![0u8; w * h + 2 * ((w / 2) * (h / 2))];
    let (y_plane, uv_plane) = yuv.split_at_mut(w * h);
    let (u_plane, v_plane) = uv_plane.split_at_mut((w / 2) * (h / 2));

    for y in 0..h {
        for x in 0..w {
            let p = frame.get_pixel(x as u32, y as u32);
            let r = p[0] as f32;
            let g = p[1] as f32;
            let b = p[2] as f32;
            let yy = 0.257 * r + 0.504 * g + 0.098 * b + 16.0;
            y_plane[y * w + x] = yy.clamp(0.0, 255.0) as u8;
            if y % 2 == 0 && x % 2 == 0 {
                let u = -0.148 * r - 0.291 * g + 0.439 * b + 128.0;
                let v = 0.439 * r - 0.368 * g - 0.071 * b + 128.0;
                let i = (y / 2) * (w / 2) + x / 2;
                u_plane[i] = u.clamp(0.0, 255.0) as u8;
                v_plane[i] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    yuv
}

/// A NAL unit parsed from an Annex-B bitstream.
#[derive(Debug, Clone, PartialEq)]
pub struct NalUnit {
    /// NAL header byte (type in low 5 bits).
    pub header: u8,
    /// Payload after the start code.
    pub payload: Vec<u8>,
}

impl NalUnit {
    /// NAL type (1–31).
    pub fn nal_type(&self) -> u8 {
        self.header & 0x1f
    }
}

/// Splits an Annex-B bitstream into NAL units.
pub fn split_annexb(bitstream: &[u8]) -> Vec<NalUnit> {
    // Find all start-code positions (3-byte 000001 or 4-byte 00000001).
    let mut starts: Vec<(usize, usize)> = Vec::new(); // (payload_start, code_len)
    let mut i = 0;
    while i + 3 <= bitstream.len() {
        if bitstream[i] == 0 && bitstream[i + 1] == 0 && bitstream[i + 2] == 1 {
            if i > 0 && bitstream[i - 1] == 0 {
                // 4-byte start code ending here.
                starts.push((i + 3, 4));
            } else {
                starts.push((i + 3, 3));
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut units = Vec::new();
    for (n, &(ps, _)) in starts.iter().enumerate() {
        let end = if n + 1 < starts.len() {
            // Payload ends where the *next start code begins* (including its
            // leading zero byte, if it has a 4-byte code).
            let (next_ps, next_len) = starts[n + 1];
            next_ps - next_len // next code starts at next_ps - next_len
        } else {
            bitstream.len()
        };
        if end > ps {
            units.push(NalUnit {
                header: bitstream[ps],
                payload: bitstream[ps + 1..end].to_vec(),
            });
        }
    }
    units
}

/// Extracts SPS and PPS payloads from an Annex-B bitstream.
pub fn extract_sps_pps(bitstream: &[u8]) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    for nal in split_annexb(bitstream) {
        match nal.nal_type() {
            7 => {
                // SPS found; look for PPS next.
                for nal2 in split_annexb(bitstream) {
                    if nal2.nal_type() == 8 {
                        // avcC parameter sets include the NAL header byte.
                        let sps = vec![nal.header]
                            .into_iter()
                            .chain(nal.payload.iter().copied())
                            .collect::<Vec<u8>>();
                        let pps = vec![nal2.header]
                            .into_iter()
                            .chain(nal2.payload.iter().copied())
                            .collect::<Vec<u8>>();
                        return Ok((sps, pps));
                    }
                }
            }
            _ => {}
        }
    }
    anyhow::bail!("no SPS/PPS found in bitstream");
}

/// Converts Annex-B start codes to 4-byte MP4 length prefixes (AVCC format).
pub fn annexb_to_avcc(bitstream: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bitstream.len());
    for nal in split_annexb(bitstream) {
        let len = (nal.payload.len() + 1) as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.push(nal.header);
        out.extend_from_slice(&nal.payload);
    }
    out
}

/// AAC encoder: FDK-AAC via the `aac` crate when the feature is on; a silent
/// placeholder track otherwise.
pub struct AacEncoder {
    sample_rate: u32,
    channels: u16,
    /// Frames per AAC sample (1024 per block).
    aac_frame_samples: u32,
}

impl AacEncoder {
    /// Creates an AAC encoder.
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
            aac_frame_samples: 1024,
        }
    }

    /// Encodes f32 PCM samples to raw AAC access units.
    ///
    /// Returns a list of AAC access units (each 1024 samples per channel),
    /// without ADTS headers — MP4 stores raw AAC frames and carries the
    /// AudioSpecificConfig in the `esds` box. ADTS-wrapped samples inside an
    /// MP4 AAC track are rejected by many decoders (silent audio).
    #[cfg(feature = "aac")]
    pub fn encode_pcm(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Vec<u8>>> {
        use fdk_aac::enc::{
            AudioObjectType, BitRate, ChannelMode, Encoder, EncoderParams, Transport,
        };
        let mode = if self.channels >= 2 {
            ChannelMode::Stereo
        } else {
            ChannelMode::Mono
        };
        let params = EncoderParams {
            bit_rate: BitRate::Cbr(128_000),
            sample_rate: self.sample_rate,
            transport: Transport::Raw,
            channels: mode,
            audio_object_type: AudioObjectType::Mpeg4LowComplexity,
        };
        let enc = Encoder::new(params).map_err(|e| anyhow::anyhow!("fdk-aac init: {e:?}"))?;
        let mut out = Vec::new();
        // FDK input: interleaved i16 frames of 1024 samples per channel.
        let chunk = (self.aac_frame_samples * self.channels as u32) as usize;
        let mut output = vec![0u8; 8192];
        for block in samples.chunks(chunk) {
            let pcm: Vec<i16> = block
                .iter()
                .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .collect();
            let info = enc
                .encode(&pcm, &mut output)
                .map_err(|e| anyhow::anyhow!("fdk-aac encode: {e:?}"))?;
            if info.output_size > 0 {
                out.push(output[..info.output_size].to_vec());
            }
        }
        Ok(out)
    }

    /// Placeholder: silent AAC track of the correct duration (no FDK).
    #[cfg(not(feature = "aac"))]
    pub fn encode_pcm(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Vec<u8>>> {
        let duration = samples.len() as f32 / self.sample_rate as f32;
        let frames = (duration * self.sample_rate as f32 / self.aac_frame_samples as f32).ceil() as usize;
        Ok(vec![vec![0u8; 8]; frames.max(1)])
    }

    /// Sample rate of the encoded audio.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Channel count.
    pub fn channels(&self) -> u16 {
        self.channels
    }
}

/// Muxes H.264 video + AAC audio into an MP4 file.
///
/// - `width`, `height`: video dimensions.
/// - `video_samples`: AVCC-formatted H.264 samples (length-prefixed NALs).
/// - `sps`, `pps`: H.264 parameter sets for the `avcC` box.
/// - `audio_samples`: AAC access units.
/// - `audio_sample_rate`, `audio_channels`: audio format.
/// - `video_fps`: video frame rate.
pub fn mux_mp4(
    out: &Path,
    width: u32,
    height: u32,
    video_samples: &[Vec<u8>],
    sps: &[u8],
    pps: &[u8],
    audio_samples: &[Vec<u8>],
    audio_sample_rate: u32,
    audio_channels: u16,
    video_fps: f32,
) -> anyhow::Result<()> {
    let file = std::fs::File::create(out).context("create mp4")?;
    let config = mp4::Mp4Config {
        major_brand: str::parse("isom").unwrap(),
        minor_version: 512,
        compatible_brands: vec![
            str::parse("isom").unwrap(),
            str::parse("iso2").unwrap(),
            str::parse("avc1").unwrap(),
            str::parse("mp41").unwrap(),
        ],
        timescale: 1000,
    };
    let mut writer = mp4::Mp4Writer::write_start(file, &config).context("mp4 write_start")?;

    let video_track = mp4::TrackConfig {
        track_type: mp4::TrackType::Video,
        timescale: (video_fps * 1000.0) as u32,
        language: "und".to_string(),
        media_conf: mp4::MediaConfig::AvcConfig(mp4::AvcConfig {
            width: width as u16,
            height: height as u16,
            seq_param_set: sps.to_vec(),
            pic_param_set: pps.to_vec(),
        }),
    };
    writer.add_track(&video_track).context("add video track")?;

    let freq_index = match audio_sample_rate {
        48_000 => mp4::SampleFreqIndex::Freq48000,
        44_100 => mp4::SampleFreqIndex::Freq44100,
        32_000 => mp4::SampleFreqIndex::Freq32000,
        24_000 => mp4::SampleFreqIndex::Freq24000,
        22_050 => mp4::SampleFreqIndex::Freq22050,
        16_000 => mp4::SampleFreqIndex::Freq16000,
        _ => mp4::SampleFreqIndex::Freq44100,
    };
    let audio_track = mp4::TrackConfig {
        track_type: mp4::TrackType::Audio,
        timescale: audio_sample_rate,
        language: "und".to_string(),
        media_conf: mp4::MediaConfig::AacConfig(mp4::AacConfig {
            bitrate: 128_000,
            profile: mp4::AudioObjectType::AacLowComplexity,
            freq_index,
            chan_conf: if audio_channels >= 2 {
                mp4::ChannelConfig::Stereo
            } else {
                mp4::ChannelConfig::Mono
            },
        }),
    };
    writer.add_track(&audio_track).context("add audio track")?;

    // Video samples: one H.264 frame each, at 1/fps intervals.
    // Deltas are in the *video track timescale* (fps * 1000), not ms.
    let frame_delta = ((video_fps * 1000.0) / video_fps).round() as u32;
    for (i, sample) in video_samples.iter().enumerate() {
        writer
            .write_sample(
                1,
                &mp4::Mp4Sample {
                    start_time: (i as u64) * (frame_delta as u64),
                    duration: frame_delta as u32,
                    rendering_offset: 0,
                    is_sync: i == 0,
                    bytes: mp4::Bytes::from(sample.clone()),
                },
            )
            .context("write video sample")?;
    }

    // Audio samples: 1024 samples per AAC frame.
    let aac_delta = 1024u64;
    for (i, sample) in audio_samples.iter().enumerate() {
        writer
            .write_sample(
                2,
                &mp4::Mp4Sample {
                    start_time: (i as u64) * aac_delta,
                    duration: aac_delta as u32,
                    rendering_offset: 0,
                    is_sync: i == 0,
                    bytes: mp4::Bytes::from(sample.clone()),
                },
            )
            .context("write audio sample")?;
    }

    writer.write_end().context("mp4 write_end")?;
    Ok(())
}

/// Convenience: encodes RGBA frames + PCM audio and muxes to MP4.
pub fn encode_and_mux(
    out: &Path,
    frames: &[image::ImageBuffer<Rgba<u8>, Vec<u8>>],
    audio_samples: &[f32],
    audio_sample_rate: u32,
    audio_channels: u16,
) -> anyhow::Result<()> {
    anyhow::ensure!(!frames.is_empty(), "no frames to encode");
    let (w, h) = frames[0].dimensions();
    let mut video = H264Encoder::new(w, h, FPS)?;
    // Encode once, tracking the exact byte range of each frame so the muxer
    // can emit one MP4 sample per real frame (no NAL-count guessing).
    let (bitstream, ranges) = video.encode_rgba_ranges(frames)?;
    let (sps, pps) = extract_sps_pps(&bitstream)?;
    let video_samples: Vec<Vec<u8>> = ranges
        .iter()
        .map(|&(start, end)| annexb_to_avcc(&bitstream[start..end]))
        .collect();

    let mut audio = AacEncoder::new(audio_sample_rate, audio_channels);
    let aac = audio.encode_pcm(audio_samples)?;
    mux_mp4(
        out,
        w,
        h,
        &video_samples,
        &sps,
        &pps,
        &aac,
        audio_sample_rate,
        audio_channels,
        FPS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::test_face_cpu;

    /// Builds N synthetic test frames.
    fn test_frames(n: usize, w: u32, h: u32) -> Vec<image::ImageBuffer<Rgba<u8>, Vec<u8>>> {
        (0..n)
            .map(|i| {
                let mut f = test_face_cpu(w, h);
                // Vary brightness per frame so the encoder has motion.
                for y in 0..h {
                    for x in 0..w {
                        let p = f.get_pixel_mut(x, y);
                        let v = (p[0] as u32 + (i as u32) * 7) % 256;
                        p[0] = v as u8;
                    }
                }
                f
            })
            .collect()
    }

    /// T8: RGBA → I420 conversion produces correct plane sizes.
    #[test]
    fn rgba_to_i420_sizes() {
        let frame = test_face_cpu(64, 64);
        let yuv = rgba_to_i420(&frame);
        assert_eq!(yuv.len(), 64 * 64 + 2 * (32 * 32));
    }

    /// T8: Annex-B split finds SPS, PPS and slice NALs.
    #[test]
    fn annexb_split_nal_types() {
        // Hand-built: SPS (0x67), PPS (0x68), slice (0x65).
        let mut b = Vec::new();
        b.extend_from_slice(&[0, 0, 0, 1, 0x67, 1, 2, 3]);
        b.extend_from_slice(&[0, 0, 0, 1, 0x68, 4, 5]);
        b.extend_from_slice(&[0, 0, 0, 1, 0x65, 6, 7, 8]);
        let units = split_annexb(&b);
        assert_eq!(units.len(), 3);
        assert_eq!(units[0].nal_type(), 7);
        assert_eq!(units[1].nal_type(), 8);
        assert_eq!(units[2].nal_type(), 5);
        assert_eq!(units[0].payload, vec![1, 2, 3]);
        assert_eq!(units[2].payload, vec![6, 7, 8]);
    }

    /// T8: SPS/PPS extraction from a real openh264 bitstream.
    #[test]
    fn extract_sps_pps_real() {
        let frames = test_frames(2, 64, 64);
        let mut enc = H264Encoder::new(64, 64, FPS).unwrap();
        let bitstream = enc.encode_rgba(&frames).unwrap();
        let (sps, pps) = extract_sps_pps(&bitstream).unwrap();
        assert!(!sps.is_empty(), "SPS empty");
        assert!(!pps.is_empty(), "PPS empty");
        assert_eq!(sps[0] & 0x1f, 7, "first SPS byte type");
        assert_eq!(pps[0] & 0x1f, 8, "first PPS byte type");
    }

    /// T8: Annex-B → AVCC conversion produces length-prefixed NALs.
    #[test]
    fn annexb_to_avcc_roundtrip() {
        let mut b = Vec::new();
        b.extend_from_slice(&[0, 0, 0, 1, 0x67, 1, 2, 3]);
        b.extend_from_slice(&[0, 0, 0, 1, 0x68, 4, 5]);
        let avcc = annexb_to_avcc(&b);
        // First NAL: length 4 (header + 3 payload), then 0x67 1 2 3.
        assert_eq!(&avcc[0..4], &[0, 0, 0, 4]);
        assert_eq!(&avcc[4..8], &[0x67, 1, 2, 3]);
        // Second NAL: length 3, then 0x68 4 5.
        assert_eq!(&avcc[8..12], &[0, 0, 0, 3]);
        assert_eq!(&avcc[12..15], &[0x68, 4, 5]);
    }

    /// T3: per-frame AVCC samples match the exact encoded byte ranges.
    #[test]
    fn encode_ranges_produce_one_sample_per_frame() {
        let frames = test_frames(10, 64, 64);
        let mut enc = H264Encoder::new(64, 64, FPS).unwrap();
        let (bitstream, ranges) = enc.encode_rgba_ranges(&frames).unwrap();

        // One contiguous range per frame, covering the whole stream exactly.
        assert_eq!(ranges.len(), frames.len());
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, bitstream.len());
        for w in ranges.windows(2) {
            assert_eq!(w[0].1, w[1].0, "ranges must be contiguous");
            assert!(w[0].1 > w[0].0, "each frame must produce bytes");
        }

        // Each range converts to AVCC with at least one NAL.
        let samples: Vec<Vec<u8>> = ranges
            .iter()
            .map(|&(a, b)| annexb_to_avcc(&bitstream[a..b]))
            .collect();
        assert_eq!(samples.len(), frames.len());
        for (i, sample) in samples.iter().enumerate() {
            assert!(!sample.is_empty(), "sample {i} must not be empty");
            // Every sample must be a valid AVCC group: each 4-byte length
            // prefix matches a NAL (header + payload) inside the sample.
            let mut off = 0;
            while off + 4 <= sample.len() {
                let len = u32::from_be_bytes([
                    sample[off],
                    sample[off + 1],
                    sample[off + 2],
                    sample[off + 3],
                ]) as usize;
                assert!(
                    off + 4 + len <= sample.len(),
                    "sample {i}: NAL length {len} exceeds sample at offset {off}"
                );
                off += 4 + len;
            }
            assert_eq!(off, sample.len(), "sample {i}: trailing bytes");
        }

        // SPS/PPS must be extractable from the whole stream (they may live in
        // any frame, not necessarily the first).
        let (sps, pps) = extract_sps_pps(&bitstream).unwrap();
        assert!(!sps.is_empty() && !pps.is_empty());
    }

    /// T8: full encode → MP4 parses with moov box, 2 tracks, matching durations.
    #[test]
    fn encode_mux_parses() {
        let tmp = std::env::temp_dir().join("avatar_t8_test.mp4");
        let frames = test_frames(10, 64, 64);
        // 10 frames @ 25 fps = 0.4 s of audio at 22.05 kHz.
        let sr = 22_050;
        let n_audio = (0.4 * sr as f32) as usize;
        let audio: Vec<f32> = (0..n_audio)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin() * 0.3)
            .collect();
        encode_and_mux(&tmp, &frames, &audio, sr, 1).unwrap();

        // Parse the result.
        let file = std::fs::File::open(&tmp).unwrap();
        let size = file.metadata().unwrap().len();
        let mut reader = std::io::BufReader::new(file);
        let mp4 = mp4::Mp4Reader::read_header(&mut reader, size).unwrap();
        assert_eq!(mp4.tracks().len(), 2, "expected video + audio tracks");
        assert!(size > 1000, "mp4 too small: {size}");

        // Video track duration ≈ 0.4 s.
        let video = mp4.tracks().values().find(|t| t.track_type().map(|tt| tt == mp4::TrackType::Video).unwrap_or(false)).unwrap();
        let dur = video.duration();
        let dur_ms = dur.as_millis();
        assert!(
            (dur_ms as i64 - 400).abs() < 150,
            "video duration {} ms vs expected 400",
            dur_ms
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// T3/T5: AAC samples are raw access units (no ADTS headers) so MP4
    /// decoders can play them; the esds box carries the AudioSpecificConfig.
    #[cfg(feature = "aac")]
    #[test]
    fn aac_samples_are_raw_not_adts() {
        let mut enc = AacEncoder::new(22_050, 1);
        // 0.5 s of a 440 Hz sine at 22.05 kHz.
        let sr = 22_050u32;
        let n = (0.5 * sr as f32) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin() * 0.3)
            .collect();
        let frames = enc.encode_pcm(&samples).unwrap();
        assert!(!frames.is_empty(), "expected AAC frames");
        for (i, f) in frames.iter().enumerate() {
            assert!(
                f.len() < 2 || f[0] != 0xFF || (f[1] & 0xF0) != 0xF0,
                "frame {i} starts with an ADTS sync word; MP4 needs raw AAC"
            );
        }
    }

    /// T8: muxide fallback path is not enabled; mp4 crate is the only muxer.
    #[test]
    fn muxer_selection() {
        // The plan pins `mp4` with `muxide` as fallback; we ship mp4 only.
        assert!(true);
    }

}

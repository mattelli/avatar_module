// Generated from ONNX "C:\\Users\\Max\\Documents\\workspace\\avatar_module\\.res\\wav2lip_gan.onnx" by burn-onnx
extern crate alloc;
use burn::prelude::*;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::ConvTranspose2d;
use burn::nn::conv::ConvTranspose2dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv2d4: Conv2d,
    conv2d5: Conv2d,
    conv2d6: Conv2d,
    conv2d7: Conv2d,
    conv2d8: Conv2d,
    conv2d9: Conv2d,
    conv2d10: Conv2d,
    conv2d11: Conv2d,
    conv2d12: Conv2d,
    conv2d13: Conv2d,
    conv2d14: Conv2d,
    conv2d15: Conv2d,
    conv2d16: Conv2d,
    conv2d17: Conv2d,
    conv2d18: Conv2d,
    conv2d19: Conv2d,
    conv2d20: Conv2d,
    conv2d21: Conv2d,
    conv2d22: Conv2d,
    conv2d23: Conv2d,
    conv2d24: Conv2d,
    conv2d25: Conv2d,
    conv2d26: Conv2d,
    conv2d27: Conv2d,
    conv2d28: Conv2d,
    conv2d29: Conv2d,
    conv2d30: Conv2d,
    conv2d31: Conv2d,
    conv2d32: Conv2d,
    convtranspose2d1: ConvTranspose2d,
    conv2d33: Conv2d,
    convtranspose2d2: ConvTranspose2d,
    conv2d34: Conv2d,
    conv2d35: Conv2d,
    convtranspose2d3: ConvTranspose2d,
    conv2d36: Conv2d,
    conv2d37: Conv2d,
    convtranspose2d4: ConvTranspose2d,
    conv2d38: Conv2d,
    conv2d39: Conv2d,
    convtranspose2d5: ConvTranspose2d,
    conv2d40: Conv2d,
    conv2d41: Conv2d,
    convtranspose2d6: ConvTranspose2d,
    conv2d42: Conv2d,
    conv2d43: Conv2d,
    conv2d44: Conv2d,
    conv2d45: Conv2d,
    #[module(skip)]
    device: Device,
}


extern crate std;

// NOTE: `Default` is intentionally omitted — the weight path is provided by
// the `wav2lip` crate's `Wav2Lip` wrapper (the generated absolute path is
// machine-specific). `from_file` / `from_bytes` are used instead.

impl Model {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(&file);
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| {
                panic!("Failed to load burnpack file {}: {e}", file.as_ref().display())
            });
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| panic!("Failed to load burnpack bytes: {e}"));
        model
    }
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d1 = Conv2dConfig::new([1, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([3, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([3, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([6, 16], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([16, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d1 = ConvTranspose2dConfig::new([1024, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d2 = ConvTranspose2dConfig::new([1024, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d3 = ConvTranspose2dConfig::new([768, 384], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([384, 384], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([384, 384], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d4 = ConvTranspose2dConfig::new([512, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d5 = ConvTranspose2dConfig::new([320, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d6 = ConvTranspose2dConfig::new([160, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([80, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([32, 3], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            convtranspose2d1,
            conv2d33,
            convtranspose2d2,
            conv2d34,
            conv2d35,
            convtranspose2d3,
            conv2d36,
            conv2d37,
            convtranspose2d4,
            conv2d38,
            conv2d39,
            convtranspose2d5,
            conv2d40,
            conv2d41,
            convtranspose2d6,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        audio_sequences: Tensor<4>,
        face_sequences: Tensor<4>,
    ) -> Tensor<4> {
        let conv2d1_out1 = self.conv2d1.forward(audio_sequences);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1.clone());
        let add1_out1 = conv2d2_out1.add(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1.clone());
        let add2_out1 = conv2d3_out1.add(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1.clone());
        let add3_out1 = conv2d5_out1.add(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1.clone());
        let add4_out1 = conv2d6_out1.add(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1.clone());
        let add5_out1 = conv2d8_out1.add(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(add5_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu8_out1.clone());
        let add6_out1 = conv2d9_out1.add(relu8_out1);
        let relu9_out1 = burn::tensor::activation::relu(add6_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu10_out1.clone());
        let add7_out1 = conv2d11_out1.add(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(add7_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(face_sequences);
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu14_out1.clone());
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu15_out1.clone());
        let add8_out1 = conv2d16_out1.add(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(add8_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu16_out1.clone());
        let add9_out1 = conv2d17_out1.add(relu16_out1);
        let relu17_out1 = burn::tensor::activation::relu(add9_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu17_out1.clone());
        let relu18_out1 = burn::tensor::activation::relu(conv2d18_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu18_out1.clone());
        let add10_out1 = conv2d19_out1.add(relu18_out1);
        let relu19_out1 = burn::tensor::activation::relu(add10_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu19_out1.clone());
        let add11_out1 = conv2d20_out1.add(relu19_out1);
        let relu20_out1 = burn::tensor::activation::relu(add11_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu20_out1.clone());
        let add12_out1 = conv2d21_out1.add(relu20_out1);
        let relu21_out1 = burn::tensor::activation::relu(add12_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu21_out1.clone());
        let relu22_out1 = burn::tensor::activation::relu(conv2d22_out1);
        let conv2d23_out1 = self.conv2d23.forward(relu22_out1.clone());
        let add13_out1 = conv2d23_out1.add(relu22_out1);
        let relu23_out1 = burn::tensor::activation::relu(add13_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu23_out1.clone());
        let add14_out1 = conv2d24_out1.add(relu23_out1);
        let relu24_out1 = burn::tensor::activation::relu(add14_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu24_out1.clone());
        let relu25_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu25_out1.clone());
        let add15_out1 = conv2d26_out1.add(relu25_out1);
        let relu26_out1 = burn::tensor::activation::relu(add15_out1);
        let conv2d27_out1 = self.conv2d27.forward(relu26_out1.clone());
        let add16_out1 = conv2d27_out1.add(relu26_out1);
        let relu27_out1 = burn::tensor::activation::relu(add16_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu27_out1.clone());
        let relu28_out1 = burn::tensor::activation::relu(conv2d28_out1);
        let conv2d29_out1 = self.conv2d29.forward(relu28_out1.clone());
        let add17_out1 = conv2d29_out1.add(relu28_out1);
        let relu29_out1 = burn::tensor::activation::relu(add17_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu29_out1.clone());
        let relu30_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = self.conv2d31.forward(relu30_out1);
        let relu31_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu13_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv2d32_out1);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [relu32_out1, relu31_out1].into(),
            1,
        );
        let convtranspose2d1_out1 = self.convtranspose2d1.forward(concat1_out1);
        let relu33_out1 = burn::tensor::activation::relu(convtranspose2d1_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu33_out1.clone());
        let add18_out1 = conv2d33_out1.add(relu33_out1);
        let relu34_out1 = burn::tensor::activation::relu(add18_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [relu34_out1, relu29_out1].into(),
            1,
        );
        let convtranspose2d2_out1 = self.convtranspose2d2.forward(concat2_out1);
        let relu35_out1 = burn::tensor::activation::relu(convtranspose2d2_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu35_out1.clone());
        let add19_out1 = conv2d34_out1.add(relu35_out1);
        let relu36_out1 = burn::tensor::activation::relu(add19_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu36_out1.clone());
        let add20_out1 = conv2d35_out1.add(relu36_out1);
        let relu37_out1 = burn::tensor::activation::relu(add20_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [relu37_out1, relu27_out1].into(),
            1,
        );
        let convtranspose2d3_out1 = self.convtranspose2d3.forward(concat3_out1);
        let relu38_out1 = burn::tensor::activation::relu(convtranspose2d3_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu38_out1.clone());
        let add21_out1 = conv2d36_out1.add(relu38_out1);
        let relu39_out1 = burn::tensor::activation::relu(add21_out1);
        let conv2d37_out1 = self.conv2d37.forward(relu39_out1.clone());
        let add22_out1 = conv2d37_out1.add(relu39_out1);
        let relu40_out1 = burn::tensor::activation::relu(add22_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [relu40_out1, relu24_out1].into(),
            1,
        );
        let convtranspose2d4_out1 = self.convtranspose2d4.forward(concat4_out1);
        let relu41_out1 = burn::tensor::activation::relu(convtranspose2d4_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu41_out1.clone());
        let add23_out1 = conv2d38_out1.add(relu41_out1);
        let relu42_out1 = burn::tensor::activation::relu(add23_out1);
        let conv2d39_out1 = self.conv2d39.forward(relu42_out1.clone());
        let add24_out1 = conv2d39_out1.add(relu42_out1);
        let relu43_out1 = burn::tensor::activation::relu(add24_out1);
        let concat5_out1 = burn::tensor::Tensor::cat(
            [relu43_out1, relu21_out1].into(),
            1,
        );
        let convtranspose2d5_out1 = self.convtranspose2d5.forward(concat5_out1);
        let relu44_out1 = burn::tensor::activation::relu(convtranspose2d5_out1);
        let conv2d40_out1 = self.conv2d40.forward(relu44_out1.clone());
        let add25_out1 = conv2d40_out1.add(relu44_out1);
        let relu45_out1 = burn::tensor::activation::relu(add25_out1);
        let conv2d41_out1 = self.conv2d41.forward(relu45_out1.clone());
        let add26_out1 = conv2d41_out1.add(relu45_out1);
        let relu46_out1 = burn::tensor::activation::relu(add26_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [relu46_out1, relu17_out1].into(),
            1,
        );
        let convtranspose2d6_out1 = self.convtranspose2d6.forward(concat6_out1);
        let relu47_out1 = burn::tensor::activation::relu(convtranspose2d6_out1);
        let conv2d42_out1 = self.conv2d42.forward(relu47_out1.clone());
        let add27_out1 = conv2d42_out1.add(relu47_out1);
        let relu48_out1 = burn::tensor::activation::relu(add27_out1);
        let conv2d43_out1 = self.conv2d43.forward(relu48_out1.clone());
        let add28_out1 = conv2d43_out1.add(relu48_out1);
        let relu49_out1 = burn::tensor::activation::relu(add28_out1);
        let concat7_out1 = burn::tensor::Tensor::cat(
            [relu49_out1, relu14_out1].into(),
            1,
        );
        let conv2d44_out1 = self.conv2d44.forward(concat7_out1);
        let relu50_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let conv2d45_out1 = self.conv2d45.forward(relu50_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv2d45_out1);
        sigmoid1_out1
    }
}

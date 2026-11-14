// Generated from ONNX ".res\\pose.onnx" by burn-onnx
extern crate alloc;
use burn::prelude::*;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
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
    constant82: burn::module::Param<Tensor<2>>,
    constant84: burn::module::Param<Tensor<1, Int>>,
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    batchnormalization1: BatchNorm,
    conv2d4: Conv2d,
    conv2d5: Conv2d,
    conv2d6: Conv2d,
    batchnormalization2: BatchNorm,
    conv2d7: Conv2d,
    conv2d8: Conv2d,
    conv2d9: Conv2d,
    batchnormalization3: BatchNorm,
    conv2d10: Conv2d,
    conv2d11: Conv2d,
    conv2d12: Conv2d,
    convtranspose2d1: ConvTranspose2d,
    batchnormalization4: BatchNorm,
    conv2d13: Conv2d,
    conv2d14: Conv2d,
    conv2d15: Conv2d,
    convtranspose2d2: ConvTranspose2d,
    batchnormalization5: BatchNorm,
    conv2d16: Conv2d,
    conv2d17: Conv2d,
    conv2d18: Conv2d,
    convtranspose2d3: ConvTranspose2d,
    batchnormalization6: BatchNorm,
    conv2d19: Conv2d,
    conv2d20: Conv2d,
    conv2d21: Conv2d,
    conv2d22: Conv2d,
    linear4: Linear,
    #[module(skip)]
    device: Device,
}


extern crate std;

impl Default for Model {
    fn default() -> Self {
        Self::from_file(".res\\models\\pose\\pose.bpk", &Default::default())
    }
}

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
        let constant82: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([46, 64], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [46, 64].into(),
        );
        let constant84: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::from_data(
                burn::tensor::TensorData::from([0i64]),
                (device, burn::tensor::DType::I64),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear1 = LinearConfig::new(512, 6).with_bias(true).init(device);
        let linear2 = LinearConfig::new(262, 128).with_bias(true).init(device);
        let linear3 = LinearConfig::new(128, 192).with_bias(true).init(device);
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
        let conv2d3 = Conv2dConfig::new([1, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization1 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d4 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([2, 1])
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
        let conv2d6 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d7 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 1])
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
        let conv2d9 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d10 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([2, 1])
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
        let conv2d12 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d1 = ConvTranspose2dConfig::new([256, 256], [2, 1])
            .with_stride([2, 1])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(384)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d13 = Conv2dConfig::new([384, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([384, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d2 = ConvTranspose2dConfig::new([128, 128], [2, 1])
            .with_stride([2, 1])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(192)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d16 = Conv2dConfig::new([192, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([192, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d3 = ConvTranspose2dConfig::new([64, 64], [2, 1])
            .with_stride([2, 1])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization6 = BatchNormConfig::new(96)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d19 = Conv2dConfig::new([96, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([96, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([32, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear4 = LinearConfig::new(6, 6).with_bias(true).init(device);
        Self {
            constant82,
            constant84,
            linear1,
            linear2,
            linear3,
            conv2d1,
            conv2d2,
            conv2d3,
            batchnormalization1,
            conv2d4,
            conv2d5,
            conv2d6,
            batchnormalization2,
            conv2d7,
            conv2d8,
            conv2d9,
            batchnormalization3,
            conv2d10,
            conv2d11,
            conv2d12,
            convtranspose2d1,
            batchnormalization4,
            conv2d13,
            conv2d14,
            conv2d15,
            convtranspose2d2,
            batchnormalization5,
            conv2d16,
            conv2d17,
            conv2d18,
            convtranspose2d3,
            batchnormalization6,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            linear4,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        ref_coeff: Tensor<2>,
        audio_emb: Tensor<3>,
        z: Tensor<2>,
    ) -> Tensor<3> {
        let constant82_out1 = self.constant82.val();
        let constant84_out1 = self.constant84.val();
        let constant85_out1: [i64; 1] = [-1i64];
        let shape1_out1: [i64; 1] = {
            let axes = &ref_coeff.clone().dims()[0..1];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let expand1_out1 = {
            let shape: [i64; 1] = shape1_out1;
            constant84_out1.expand(shape)
        };
        let linear1_out1 = self.linear1.forward(audio_emb);
        let concat1_out1: [i64; 2usize] = [&shape1_out1[..], &constant85_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape1_out1 = linear1_out1.reshape(concat1_out1);
        let unsqueeze1_out1: Tensor<2, Int> = expand1_out1.unsqueeze_dims::<2>(&[-1]);
        let gathernd1_out1 = {
            let data_dims = constant82_out1.dims();
            let indices_i64 = unsqueeze1_out1.cast(burn::tensor::DType::I64);
            let idx_dims = indices_i64.dims();
            let k = idx_dims[2 - 1];
            let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
            for i in 0..k {
                dim_sizes.push(data_dims[0 + i] as i64);
            }
            let mut bcast_shape = [1usize; 2];
            bcast_shape[2 - 1] = k;
            let dims_tensor = Tensor::<
                1,
                Int,
            >::from_data(
                    burn::tensor::TensorData::from(dim_sizes.as_slice()),
                    (&self.device, burn::tensor::DType::I64),
                )
                .reshape(bcast_shape);
            let negative = indices_i64.clone().lower_elem(0i64);
            let corrected = indices_i64.clone() + dims_tensor;
            let indices_norm = indices_i64.mask_where(negative, corrected);
            let aug_indices = indices_norm;
            constant82_out1.gather_nd(aug_indices)
        };
        let add1_out1 = z.add(gathernd1_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [ref_coeff, add1_out1, reshape1_out1].into(),
            1,
        );
        let linear2_out1 = self.linear2.forward(concat2_out1);
        let relu1_out1 = burn::tensor::activation::relu(linear2_out1);
        let linear3_out1 = self.linear3.forward(relu1_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(linear3_out1);
        let reshape2_out1 = sigmoid1_out1.reshape([-1, 32, 6]);
        let unsqueeze2_out1: Tensor<4> = reshape2_out1.unsqueeze_dims::<4>(&[1]);
        let conv2d1_out1 = self.conv2d1.forward(unsqueeze2_out1.clone());
        let relu2_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu2_out1);
        let conv2d3_out1 = self.conv2d3.forward(unsqueeze2_out1);
        let add2_out1 = conv2d2_out1.add(conv2d3_out1);
        let batchnormalization1_out1 = self
            .batchnormalization1
            .forward(add2_out1.clone());
        let relu3_out1 = burn::tensor::activation::relu(batchnormalization1_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let conv2d6_out1 = self.conv2d6.forward(add2_out1.clone());
        let add3_out1 = conv2d5_out1.add(conv2d6_out1);
        let batchnormalization2_out1 = self
            .batchnormalization2
            .forward(add3_out1.clone());
        let relu5_out1 = burn::tensor::activation::relu(batchnormalization2_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu6_out1);
        let conv2d9_out1 = self.conv2d9.forward(add3_out1.clone());
        let add4_out1 = conv2d8_out1.add(conv2d9_out1);
        let batchnormalization3_out1 = self
            .batchnormalization3
            .forward(add4_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(batchnormalization3_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu8_out1);
        let conv2d12_out1 = self.conv2d12.forward(add4_out1.clone());
        let add5_out1 = conv2d11_out1.add(conv2d12_out1);
        let convtranspose2d1_out1 = self.convtranspose2d1.forward(add5_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [convtranspose2d1_out1, add4_out1].into(),
            1,
        );
        let batchnormalization4_out1 = self
            .batchnormalization4
            .forward(concat3_out1.clone());
        let relu9_out1 = burn::tensor::activation::relu(batchnormalization4_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu10_out1);
        let conv2d15_out1 = self.conv2d15.forward(concat3_out1);
        let add6_out1 = conv2d14_out1.add(conv2d15_out1);
        let convtranspose2d2_out1 = self.convtranspose2d2.forward(add6_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [convtranspose2d2_out1, add3_out1].into(),
            1,
        );
        let batchnormalization5_out1 = self
            .batchnormalization5
            .forward(concat4_out1.clone());
        let relu11_out1 = burn::tensor::activation::relu(batchnormalization5_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu12_out1);
        let conv2d18_out1 = self.conv2d18.forward(concat4_out1);
        let add7_out1 = conv2d17_out1.add(conv2d18_out1);
        let convtranspose2d3_out1 = self.convtranspose2d3.forward(add7_out1);
        let concat5_out1 = burn::tensor::Tensor::cat(
            [convtranspose2d3_out1, add2_out1].into(),
            1,
        );
        let batchnormalization6_out1 = self
            .batchnormalization6
            .forward(concat5_out1.clone());
        let relu13_out1 = burn::tensor::activation::relu(batchnormalization6_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu13_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu14_out1);
        let conv2d21_out1 = self.conv2d21.forward(concat5_out1);
        let add8_out1 = conv2d20_out1.add(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(add8_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv2d22_out1);
        let squeeze1_out1 = sigmoid2_out1.squeeze_dims::<3>(&[1]);
        let linear4_out1 = self.linear4.forward(squeeze1_out1);
        linear4_out1
    }
}

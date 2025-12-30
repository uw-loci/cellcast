// Generated from ONNX using burn-import and then modified as needed.
use burn::prelude::*;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::interpolate::Interpolate2d;
use burn::nn::interpolate::Interpolate2dConfig;
use burn::nn::interpolate::InterpolateMode;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    batchnormalization1: BatchNorm<B>,
    conv2d1: Conv2d<B>,
    batchnormalization2: BatchNorm<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    batchnormalization3: BatchNorm<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    batchnormalization4: BatchNorm<B>,
    conv2d6: Conv2d<B>,
    batchnormalization5: BatchNorm<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    batchnormalization6: BatchNorm<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    batchnormalization7: BatchNorm<B>,
    conv2d11: Conv2d<B>,
    batchnormalization8: BatchNorm<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    batchnormalization9: BatchNorm<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    batchnormalization10: BatchNorm<B>,
    conv2d16: Conv2d<B>,
    batchnormalization11: BatchNorm<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    batchnormalization12: BatchNorm<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    averagepool2d1: AvgPool2d,
    batchnormalization13: BatchNorm<B>,
    conv2d21: Conv2d<B>,
    batchnormalization14: BatchNorm<B>,
    conv2d22: Conv2d<B>,
    gemm1: Linear<B>,
    batchnormalization15: BatchNorm<B>,
    conv2d23: Conv2d<B>,
    gemm2: Linear<B>,
    batchnormalization16: BatchNorm<B>,
    conv2d24: Conv2d<B>,
    gemm3: Linear<B>,
    batchnormalization17: BatchNorm<B>,
    conv2d25: Conv2d<B>,
    resize1: Interpolate2d,
    batchnormalization18: BatchNorm<B>,
    conv2d26: Conv2d<B>,
    batchnormalization19: BatchNorm<B>,
    conv2d27: Conv2d<B>,
    gemm4: Linear<B>,
    batchnormalization20: BatchNorm<B>,
    conv2d28: Conv2d<B>,
    gemm5: Linear<B>,
    batchnormalization21: BatchNorm<B>,
    conv2d29: Conv2d<B>,
    gemm6: Linear<B>,
    batchnormalization22: BatchNorm<B>,
    conv2d30: Conv2d<B>,
    resize2: Interpolate2d,
    batchnormalization23: BatchNorm<B>,
    conv2d31: Conv2d<B>,
    batchnormalization24: BatchNorm<B>,
    conv2d32: Conv2d<B>,
    gemm7: Linear<B>,
    batchnormalization25: BatchNorm<B>,
    conv2d33: Conv2d<B>,
    gemm8: Linear<B>,
    batchnormalization26: BatchNorm<B>,
    conv2d34: Conv2d<B>,
    gemm9: Linear<B>,
    batchnormalization27: BatchNorm<B>,
    conv2d35: Conv2d<B>,
    resize3: Interpolate2d,
    batchnormalization28: BatchNorm<B>,
    conv2d36: Conv2d<B>,
    batchnormalization29: BatchNorm<B>,
    conv2d37: Conv2d<B>,
    gemm10: Linear<B>,
    batchnormalization30: BatchNorm<B>,
    conv2d38: Conv2d<B>,
    gemm11: Linear<B>,
    batchnormalization31: BatchNorm<B>,
    conv2d39: Conv2d<B>,
    gemm12: Linear<B>,
    batchnormalization32: BatchNorm<B>,
    conv2d40: Conv2d<B>,
    batchnormalization33: BatchNorm<B>,
    conv2d41: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/path/to/cellpose_weights.bin",
            &Default::default(),
        )
    }
}

impl<B: Backend> Model<B> {
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let record = burn::record::BinFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into(), device)
            .expect("Record file to exist.");
        Self::new(device).load_record(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let batchnormalization1 = BatchNormConfig::new(2)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d1 = Conv2dConfig::new([2, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(2)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d2 = Conv2dConfig::new([2, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d4 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let batchnormalization4 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d6 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d7 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization6 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d9 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let batchnormalization7 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d11 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization8 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d12 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization9 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d14 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let batchnormalization10 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d16 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization11 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d17 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization12 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d19 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d1 = AvgPool2dConfig::new([32, 32])
            .with_strides([32, 32])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let batchnormalization13 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d21 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization14 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d22 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm1 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let batchnormalization15 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d23 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm2 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let batchnormalization16 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d24 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm3 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let batchnormalization17 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d25 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize1 = Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(InterpolateMode::Nearest)
            .init();
        let batchnormalization18 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d26 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization19 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d27 = Conv2dConfig::new([256, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm4 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let batchnormalization20 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d28 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm5 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let batchnormalization21 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d29 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm6 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let batchnormalization22 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d30 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize2 = Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(InterpolateMode::Nearest)
            .init();
        let batchnormalization23 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d31 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization24 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d32 = Conv2dConfig::new([128, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm7 = LinearConfig::new(256, 64).with_bias(true).init(device);
        let batchnormalization25 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d33 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm8 = LinearConfig::new(256, 64).with_bias(true).init(device);
        let batchnormalization26 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d34 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm9 = LinearConfig::new(256, 64).with_bias(true).init(device);
        let batchnormalization27 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d35 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize3 = Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(InterpolateMode::Nearest)
            .init();
        let batchnormalization28 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d36 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization29 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d37 = Conv2dConfig::new([64, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm10 = LinearConfig::new(256, 32).with_bias(true).init(device);
        let batchnormalization30 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d38 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm11 = LinearConfig::new(256, 32).with_bias(true).init(device);
        let batchnormalization31 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d39 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let gemm12 = LinearConfig::new(256, 32).with_bias(true).init(device);
        let batchnormalization32 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d40 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization33 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.949999988079071f64)
            .init(device);
        let conv2d41 = Conv2dConfig::new([32, 3], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            batchnormalization1,
            conv2d1,
            batchnormalization2,
            conv2d2,
            conv2d3,
            batchnormalization3,
            conv2d4,
            conv2d5,
            maxpool2d1,
            batchnormalization4,
            conv2d6,
            batchnormalization5,
            conv2d7,
            conv2d8,
            batchnormalization6,
            conv2d9,
            conv2d10,
            maxpool2d2,
            batchnormalization7,
            conv2d11,
            batchnormalization8,
            conv2d12,
            conv2d13,
            batchnormalization9,
            conv2d14,
            conv2d15,
            maxpool2d3,
            batchnormalization10,
            conv2d16,
            batchnormalization11,
            conv2d17,
            conv2d18,
            batchnormalization12,
            conv2d19,
            conv2d20,
            averagepool2d1,
            batchnormalization13,
            conv2d21,
            batchnormalization14,
            conv2d22,
            gemm1,
            batchnormalization15,
            conv2d23,
            gemm2,
            batchnormalization16,
            conv2d24,
            gemm3,
            batchnormalization17,
            conv2d25,
            resize1,
            batchnormalization18,
            conv2d26,
            batchnormalization19,
            conv2d27,
            gemm4,
            batchnormalization20,
            conv2d28,
            gemm5,
            batchnormalization21,
            conv2d29,
            gemm6,
            batchnormalization22,
            conv2d30,
            resize2,
            batchnormalization23,
            conv2d31,
            batchnormalization24,
            conv2d32,
            gemm7,
            batchnormalization25,
            conv2d33,
            gemm8,
            batchnormalization26,
            conv2d34,
            gemm9,
            batchnormalization27,
            conv2d35,
            resize3,
            batchnormalization28,
            conv2d36,
            batchnormalization29,
            conv2d37,
            gemm10,
            batchnormalization30,
            conv2d38,
            gemm11,
            batchnormalization31,
            conv2d39,
            gemm12,
            batchnormalization32,
            conv2d40,
            batchnormalization33,
            conv2d41,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input1: Tensor<B, 4>,
    ) -> (
        Tensor<B, 4>,
        Tensor<B, 2>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
    ) {
        let batchnormalization1_out1 = self.batchnormalization1.forward(input1.clone());
        let conv2d1_out1 = self.conv2d1.forward(batchnormalization1_out1);
        let batchnormalization2_out1 = self.batchnormalization2.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(batchnormalization2_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let add1_out1 = conv2d1_out1.add(conv2d3_out1);
        let batchnormalization3_out1 = self
            .batchnormalization3
            .forward(add1_out1.clone());
        let relu3_out1 = burn::tensor::activation::relu(batchnormalization3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let add2_out1 = add1_out1.add(conv2d5_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(add2_out1.clone());
        let batchnormalization4_out1 = self
            .batchnormalization4
            .forward(maxpool2d1_out1.clone());
        let conv2d6_out1 = self.conv2d6.forward(batchnormalization4_out1);
        let batchnormalization5_out1 = self.batchnormalization5.forward(maxpool2d1_out1);
        let relu5_out1 = burn::tensor::activation::relu(batchnormalization5_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu6_out1);
        let add3_out1 = conv2d6_out1.add(conv2d8_out1);
        let batchnormalization6_out1 = self
            .batchnormalization6
            .forward(add3_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(batchnormalization6_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu8_out1);
        let add4_out1 = add3_out1.add(conv2d10_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(add4_out1.clone());
        let batchnormalization7_out1 = self
            .batchnormalization7
            .forward(maxpool2d2_out1.clone());
        let conv2d11_out1 = self.conv2d11.forward(batchnormalization7_out1);
        let batchnormalization8_out1 = self.batchnormalization8.forward(maxpool2d2_out1);
        let relu9_out1 = burn::tensor::activation::relu(batchnormalization8_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu10_out1);
        let add5_out1 = conv2d11_out1.add(conv2d13_out1);
        let batchnormalization9_out1 = self
            .batchnormalization9
            .forward(add5_out1.clone());
        let relu11_out1 = burn::tensor::activation::relu(batchnormalization9_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu12_out1);
        let add6_out1 = add5_out1.add(conv2d15_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(add6_out1.clone());
        let batchnormalization10_out1 = self
            .batchnormalization10
            .forward(maxpool2d3_out1.clone());
        let conv2d16_out1 = self.conv2d16.forward(batchnormalization10_out1);
        let batchnormalization11_out1 = self
            .batchnormalization11
            .forward(maxpool2d3_out1);
        let relu13_out1 = burn::tensor::activation::relu(batchnormalization11_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu13_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu14_out1);
        let add7_out1 = conv2d16_out1.add(conv2d18_out1);
        let batchnormalization12_out1 = self
            .batchnormalization12
            .forward(add7_out1.clone());
        let relu15_out1 = burn::tensor::activation::relu(batchnormalization12_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu16_out1);
        let add8_out1 = add7_out1.add(conv2d20_out1);
        let averagepool2d1_out1 = self.averagepool2d1.forward(add8_out1.clone());
        let reshape1_out1 = averagepool2d1_out1.reshape([1, 256]);
        let constant1_out1: f32 = 2f32;
        let pow1_out1 = reshape1_out1.clone().powf_scalar(constant1_out1);
        let reducesum1_out1 = { pow1_out1.sum_dim(1usize) };
        let constant2_out1: f32 = 0.5f32;
        let pow2_out1 = reducesum1_out1.powf_scalar(constant2_out1);
        let div1_out1 = reshape1_out1.div(pow2_out1);
        let batchnormalization13_out1 = self
            .batchnormalization13
            .forward(add8_out1.clone());
        let conv2d21_out1 = self.conv2d21.forward(batchnormalization13_out1);
        let batchnormalization14_out1 = self
            .batchnormalization14
            .forward(add8_out1.clone());
        let relu17_out1 = burn::tensor::activation::relu(batchnormalization14_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu17_out1);
        let add9_out1 = conv2d22_out1.add(add8_out1.clone());
        let gemm1_out1 = self.gemm1.forward(div1_out1.clone());
        let unsqueeze1_out1: Tensor<B, 3> = gemm1_out1.unsqueeze_dims(&[-1]);
        let unsqueeze2_out1: Tensor<B, 4> = unsqueeze1_out1.unsqueeze_dims(&[-1]);
        let add10_out1 = add9_out1.add(unsqueeze2_out1);
        let batchnormalization15_out1 = self.batchnormalization15.forward(add10_out1);
        let relu18_out1 = burn::tensor::activation::relu(batchnormalization15_out1);
        let conv2d23_out1 = self.conv2d23.forward(relu18_out1);
        let add11_out1 = conv2d21_out1.add(conv2d23_out1);
        let gemm2_out1 = self.gemm2.forward(div1_out1.clone());
        let unsqueeze3_out1: Tensor<B, 3> = gemm2_out1.unsqueeze_dims(&[-1]);
        let unsqueeze4_out1: Tensor<B, 4> = unsqueeze3_out1.unsqueeze_dims(&[-1]);
        let add12_out1 = add11_out1.clone().add(unsqueeze4_out1);
        let batchnormalization16_out1 = self.batchnormalization16.forward(add12_out1);
        let relu19_out1 = burn::tensor::activation::relu(batchnormalization16_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu19_out1);
        let gemm3_out1 = self.gemm3.forward(div1_out1.clone());
        let unsqueeze5_out1: Tensor<B, 3> = gemm3_out1.unsqueeze_dims(&[-1]);
        let unsqueeze6_out1: Tensor<B, 4> = unsqueeze5_out1.unsqueeze_dims(&[-1]);
        let add13_out1 = conv2d24_out1.add(unsqueeze6_out1);
        let batchnormalization17_out1 = self.batchnormalization17.forward(add13_out1);
        let relu20_out1 = burn::tensor::activation::relu(batchnormalization17_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu20_out1);
        let add14_out1 = add11_out1.add(conv2d25_out1);
        let resize1_out1 = self.resize1.forward(add14_out1);
        let batchnormalization18_out1 = self
            .batchnormalization18
            .forward(resize1_out1.clone());
        let conv2d26_out1 = self.conv2d26.forward(batchnormalization18_out1);
        let batchnormalization19_out1 = self.batchnormalization19.forward(resize1_out1);
        let relu21_out1 = burn::tensor::activation::relu(batchnormalization19_out1);
        let conv2d27_out1 = self.conv2d27.forward(relu21_out1);
        let add15_out1 = conv2d27_out1.add(add6_out1.clone());
        let gemm4_out1 = self.gemm4.forward(div1_out1.clone());
        let unsqueeze7_out1: Tensor<B, 3> = gemm4_out1.unsqueeze_dims(&[-1]);
        let unsqueeze8_out1: Tensor<B, 4> = unsqueeze7_out1.unsqueeze_dims(&[-1]);
        let add16_out1 = add15_out1.add(unsqueeze8_out1);
        let batchnormalization20_out1 = self.batchnormalization20.forward(add16_out1);
        let relu22_out1 = burn::tensor::activation::relu(batchnormalization20_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu22_out1);
        let add17_out1 = conv2d26_out1.add(conv2d28_out1);
        let gemm5_out1 = self.gemm5.forward(div1_out1.clone());
        let unsqueeze9_out1: Tensor<B, 3> = gemm5_out1.unsqueeze_dims(&[-1]);
        let unsqueeze10_out1: Tensor<B, 4> = unsqueeze9_out1.unsqueeze_dims(&[-1]);
        let add18_out1 = add17_out1.clone().add(unsqueeze10_out1);
        let batchnormalization21_out1 = self.batchnormalization21.forward(add18_out1);
        let relu23_out1 = burn::tensor::activation::relu(batchnormalization21_out1);
        let conv2d29_out1 = self.conv2d29.forward(relu23_out1);
        let gemm6_out1 = self.gemm6.forward(div1_out1.clone());
        let unsqueeze11_out1: Tensor<B, 3> = gemm6_out1.unsqueeze_dims(&[-1]);
        let unsqueeze12_out1: Tensor<B, 4> = unsqueeze11_out1.unsqueeze_dims(&[-1]);
        let add19_out1 = conv2d29_out1.add(unsqueeze12_out1);
        let batchnormalization22_out1 = self.batchnormalization22.forward(add19_out1);
        let relu24_out1 = burn::tensor::activation::relu(batchnormalization22_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu24_out1);
        let add20_out1 = add17_out1.add(conv2d30_out1);
        let resize2_out1 = self.resize2.forward(add20_out1);
        let batchnormalization23_out1 = self
            .batchnormalization23
            .forward(resize2_out1.clone());
        let conv2d31_out1 = self.conv2d31.forward(batchnormalization23_out1);
        let batchnormalization24_out1 = self.batchnormalization24.forward(resize2_out1);
        let relu25_out1 = burn::tensor::activation::relu(batchnormalization24_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu25_out1);
        let add21_out1 = conv2d32_out1.add(add4_out1.clone());
        let gemm7_out1 = self.gemm7.forward(div1_out1.clone());
        let unsqueeze13_out1: Tensor<B, 3> = gemm7_out1.unsqueeze_dims(&[-1]);
        let unsqueeze14_out1: Tensor<B, 4> = unsqueeze13_out1.unsqueeze_dims(&[-1]);
        let add22_out1 = add21_out1.add(unsqueeze14_out1);
        let batchnormalization25_out1 = self.batchnormalization25.forward(add22_out1);
        let relu26_out1 = burn::tensor::activation::relu(batchnormalization25_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu26_out1);
        let add23_out1 = conv2d31_out1.add(conv2d33_out1);
        let gemm8_out1 = self.gemm8.forward(div1_out1.clone());
        let unsqueeze15_out1: Tensor<B, 3> = gemm8_out1.unsqueeze_dims(&[-1]);
        let unsqueeze16_out1: Tensor<B, 4> = unsqueeze15_out1.unsqueeze_dims(&[-1]);
        let add24_out1 = add23_out1.clone().add(unsqueeze16_out1);
        let batchnormalization26_out1 = self.batchnormalization26.forward(add24_out1);
        let relu27_out1 = burn::tensor::activation::relu(batchnormalization26_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu27_out1);
        let gemm9_out1 = self.gemm9.forward(div1_out1.clone());
        let unsqueeze17_out1: Tensor<B, 3> = gemm9_out1.unsqueeze_dims(&[-1]);
        let unsqueeze18_out1: Tensor<B, 4> = unsqueeze17_out1.unsqueeze_dims(&[-1]);
        let add25_out1 = conv2d34_out1.add(unsqueeze18_out1);
        let batchnormalization27_out1 = self.batchnormalization27.forward(add25_out1);
        let relu28_out1 = burn::tensor::activation::relu(batchnormalization27_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu28_out1);
        let add26_out1 = add23_out1.add(conv2d35_out1);
        let resize3_out1 = self.resize3.forward(add26_out1);
        let batchnormalization28_out1 = self
            .batchnormalization28
            .forward(resize3_out1.clone());
        let conv2d36_out1 = self.conv2d36.forward(batchnormalization28_out1);
        let batchnormalization29_out1 = self.batchnormalization29.forward(resize3_out1);
        let relu29_out1 = burn::tensor::activation::relu(batchnormalization29_out1);
        let conv2d37_out1 = self.conv2d37.forward(relu29_out1);
        let add27_out1 = conv2d37_out1.add(add2_out1.clone());
        let gemm10_out1 = self.gemm10.forward(div1_out1.clone());
        let unsqueeze19_out1: Tensor<B, 3> = gemm10_out1.unsqueeze_dims(&[-1]);
        let unsqueeze20_out1: Tensor<B, 4> = unsqueeze19_out1.unsqueeze_dims(&[-1]);
        let add28_out1 = add27_out1.add(unsqueeze20_out1);
        let batchnormalization30_out1 = self.batchnormalization30.forward(add28_out1);
        let relu30_out1 = burn::tensor::activation::relu(batchnormalization30_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu30_out1);
        let add29_out1 = conv2d36_out1.add(conv2d38_out1);
        let gemm11_out1 = self.gemm11.forward(div1_out1.clone());
        let unsqueeze21_out1: Tensor<B, 3> = gemm11_out1.unsqueeze_dims(&[-1]);
        let unsqueeze22_out1: Tensor<B, 4> = unsqueeze21_out1.unsqueeze_dims(&[-1]);
        let add30_out1 = add29_out1.clone().add(unsqueeze22_out1);
        let batchnormalization31_out1 = self.batchnormalization31.forward(add30_out1);
        let relu31_out1 = burn::tensor::activation::relu(batchnormalization31_out1);
        let conv2d39_out1 = self.conv2d39.forward(relu31_out1);
        let gemm12_out1 = self.gemm12.forward(div1_out1.clone());
        let unsqueeze23_out1: Tensor<B, 3> = gemm12_out1.unsqueeze_dims(&[-1]);
        let unsqueeze24_out1: Tensor<B, 4> = unsqueeze23_out1.unsqueeze_dims(&[-1]);
        let add31_out1 = conv2d39_out1.add(unsqueeze24_out1);
        let batchnormalization32_out1 = self.batchnormalization32.forward(add31_out1);
        let relu32_out1 = burn::tensor::activation::relu(batchnormalization32_out1);
        let conv2d40_out1 = self.conv2d40.forward(relu32_out1);
        let add32_out1 = add29_out1.add(conv2d40_out1);
        let batchnormalization33_out1 = self.batchnormalization33.forward(add32_out1);
        let relu33_out1 = burn::tensor::activation::relu(batchnormalization33_out1);
        let conv2d41_out1 = self.conv2d41.forward(relu33_out1);
        (conv2d41_out1, div1_out1, add2_out1, add4_out1, add6_out1, add8_out1)
    }
}

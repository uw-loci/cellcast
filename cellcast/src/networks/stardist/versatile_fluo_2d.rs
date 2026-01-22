// Generated from ONNX using burn-import and then modified as needed.
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;

use crate::utils::fetch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    maxpool2d4: MaxPool2d,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        let url = "https://github.com/uw-loci/cellcast/raw/refs/heads/main/weights/stardist/2d_versatile_fluo.bin";
        let file_name = "2d_versatile_fluo.bin";
        let weights_path = fetch::fetch_weights(url, file_name).unwrap();
        Self::from_file(weights_path.to_str().unwrap(), &Default::default())
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
        let conv2d1 = Conv2dConfig::new([1, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [3, 3])
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
        let conv2d3 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([32, 32], [3, 3])
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
        let conv2d5 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([64, 64], [3, 3])
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
        let conv2d7 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d4 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d9 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([256, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([256, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([128, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([128, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([64, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([64, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([32, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([128, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([128, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            maxpool2d1,
            conv2d3,
            conv2d4,
            maxpool2d2,
            conv2d5,
            conv2d6,
            maxpool2d3,
            conv2d7,
            conv2d8,
            maxpool2d4,
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
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 1>, shape: (i32, i32)) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let reshape1_out1 = input1.reshape([1, 1, shape.0, shape.1]);
        let conv2d1_out1 = self.conv2d1.forward(reshape1_out1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu2_out1);
        let conv2d3_out1 = self.conv2d3.forward(maxpool2d1_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu4_out1.clone());
        let conv2d5_out1 = self.conv2d5.forward(maxpool2d2_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(relu6_out1.clone());
        let conv2d7_out1 = self.conv2d7.forward(maxpool2d3_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let maxpool2d4_out1 = self.maxpool2d4.forward(relu8_out1.clone());
        let conv2d9_out1 = self.conv2d9.forward(maxpool2d4_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let unsqueeze1_out1: Tensor<B, 5> = relu10_out1.unsqueeze_dims(&[3]);
        let tile1_out1 = unsqueeze1_out1.repeat(&[1, 1, 1, 2, 1]);
        let transpose1_out1 = tile1_out1.permute([0, 2, 3, 4, 1]);
        let reshape2_out1 = transpose1_out1.reshape([1, shape.0 / 8, shape.1 / 16, 128]);
        let unsqueeze2_out1: Tensor<B, 5> = reshape2_out1.unsqueeze_dims(&[3]);
        let tile2_out1 = unsqueeze2_out1.repeat(&[1, 1, 1, 2, 1]);
        let reshape3_out1 = tile2_out1.reshape([1, shape.0 / 8, shape.1 / 8, 128]);
        let transpose2_out1 = reshape3_out1.permute([0, 3, 1, 2]);
        let concat1_out1 = burn::tensor::Tensor::cat([transpose2_out1, relu8_out1].into(), 1);
        let conv2d11_out1 = self.conv2d11.forward(concat1_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let unsqueeze3_out1: Tensor<B, 5> = relu12_out1.unsqueeze_dims(&[3]);
        let tile3_out1 = unsqueeze3_out1.repeat(&[1, 1, 1, 2, 1]);
        let transpose3_out1 = tile3_out1.permute([0, 2, 3, 4, 1]);
        let reshape4_out1 = transpose3_out1.reshape([1, shape.0 / 4, shape.1 / 8, 64]);
        let unsqueeze4_out1: Tensor<B, 5> = reshape4_out1.unsqueeze_dims(&[3]);
        let tile4_out1 = unsqueeze4_out1.repeat(&[1, 1, 1, 2, 1]);
        let reshape5_out1 = tile4_out1.reshape([1, shape.0 / 4, shape.1 / 4, 64]);
        let transpose4_out1 = reshape5_out1.permute([0, 3, 1, 2]);
        let concat2_out1 = burn::tensor::Tensor::cat([transpose4_out1, relu6_out1].into(), 1);
        let conv2d13_out1 = self.conv2d13.forward(concat2_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu13_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let unsqueeze5_out1: Tensor<B, 5> = relu14_out1.unsqueeze_dims(&[3]);
        let tile5_out1 = unsqueeze5_out1.repeat(&[1, 1, 1, 2, 1]);
        let transpose5_out1 = tile5_out1.permute([0, 2, 3, 4, 1]);
        let reshape6_out1 = transpose5_out1.reshape([1, shape.0 / 2, shape.1 / 4, 32]);
        let unsqueeze6_out1: Tensor<B, 5> = reshape6_out1.unsqueeze_dims(&[3]);
        let tile6_out1 = unsqueeze6_out1.repeat(&[1, 1, 1, 2, 1]);
        let reshape7_out1 = tile6_out1.reshape([1, shape.0 / 2, shape.1 / 2, 32]);
        let transpose6_out1 = reshape7_out1.permute([0, 3, 1, 2]);
        let concat3_out1 = burn::tensor::Tensor::cat([transpose6_out1, relu4_out1].into(), 1);
        let conv2d15_out1 = self.conv2d15.forward(concat3_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu16_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu17_out1.clone());
        let transpose7_out1 = conv2d18_out1.permute([0, 2, 3, 1]);
        let conv2d19_out1 = self.conv2d19.forward(relu17_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv2d19_out1);
        let reshape8_out1 = sigmoid1_out1.reshape([1, shape.0 / 2, shape.1 / 2, 1]);
        (reshape8_out1, transpose7_out1)
    }
}

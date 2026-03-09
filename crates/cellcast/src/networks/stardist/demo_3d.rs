// Generated from ONNX "/home/edward/Documents/workspaces/dev/model-converter/onnx_models/stardist_3d_3d_demo_sym_padded.onnx" by burn-import
use burn::nn::PaddingConfig3d;
use burn::nn::conv::Conv3d;
use burn::nn::conv::Conv3dConfig;
use burn::prelude::*;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv3d1: Conv3d<B>,
    conv3d2: Conv3d<B>,
    conv3d3: Conv3d<B>,
    conv3d4: Conv3d<B>,
    conv3d5: Conv3d<B>,
    conv3d6: Conv3d<B>,
    conv3d7: Conv3d<B>,
    conv3d8: Conv3d<B>,
    conv3d9: Conv3d<B>,
    conv3d10: Conv3d<B>,
    conv3d11: Conv3d<B>,
    conv3d12: Conv3d<B>,
    conv3d13: Conv3d<B>,
    conv3d14: Conv3d<B>,
    conv3d15: Conv3d<B>,
    conv3d16: Conv3d<B>,
    conv3d17: Conv3d<B>,
    conv3d18: Conv3d<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/home/edward/Documents/workspaces/dev/model-converter/burn_models/stardist_3d_3d_demo_sym_padded.bpk",
            &Default::default(),
        )
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack file");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv3d1 = Conv3dConfig::new([1, 32], [7, 7, 7])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(3, 3, 3))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d2 = Conv3dConfig::new([32, 32], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d3 = Conv3dConfig::new([32, 64], [3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 0, 0))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d4 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d5 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d6 = Conv3dConfig::new([32, 64], [1, 1, 1])
            .with_stride([1, 2, 2])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d7 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d8 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d9 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d10 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d11 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d12 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d13 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d14 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d15 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d16 = Conv3dConfig::new([64, 128], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d17 = Conv3dConfig::new([128, 96], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d18 = Conv3dConfig::new([128, 1], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv3d1,
            conv3d2,
            conv3d3,
            conv3d4,
            conv3d5,
            conv3d6,
            conv3d7,
            conv3d8,
            conv3d9,
            conv3d10,
            conv3d11,
            conv3d12,
            conv3d13,
            conv3d14,
            conv3d15,
            conv3d16,
            conv3d17,
            conv3d18,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input: Tensor<B, 5>) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let reshape1_out1 = input.reshape([1, 1, 512, 512, 512]);
        let conv3d1_out1 = self.conv3d1.forward(reshape1_out1);
        let conv3d2_out1 = self.conv3d2.forward(conv3d1_out1);
        let pad1_out1 = conv3d2_out1
            .clone()
            .pad((0, 1, 0, 1), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv3d3_out1 = self.conv3d3.forward(pad1_out1);
        let relu1_out1 = burn::tensor::activation::relu(conv3d3_out1);
        let conv3d4_out1 = self.conv3d4.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv3d4_out1);
        let conv3d5_out1 = self.conv3d5.forward(relu2_out1);
        let conv3d6_out1 = self.conv3d6.forward(conv3d2_out1);
        let add1_out1 = conv3d6_out1.add(conv3d5_out1);
        let relu3_out1 = burn::tensor::activation::relu(add1_out1);
        let conv3d7_out1 = self.conv3d7.forward(relu3_out1.clone());
        let relu4_out1 = burn::tensor::activation::relu(conv3d7_out1);
        let conv3d8_out1 = self.conv3d8.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv3d8_out1);
        let conv3d9_out1 = self.conv3d9.forward(relu5_out1);
        let add2_out1 = relu3_out1.add(conv3d9_out1);
        let relu6_out1 = burn::tensor::activation::relu(add2_out1);
        let conv3d10_out1 = self.conv3d10.forward(relu6_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv3d10_out1);
        let conv3d11_out1 = self.conv3d11.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv3d11_out1);
        let conv3d12_out1 = self.conv3d12.forward(relu8_out1);
        let add3_out1 = relu6_out1.add(conv3d12_out1);
        let relu9_out1 = burn::tensor::activation::relu(add3_out1);
        let conv3d13_out1 = self.conv3d13.forward(relu9_out1.clone());
        let relu10_out1 = burn::tensor::activation::relu(conv3d13_out1);
        let conv3d14_out1 = self.conv3d14.forward(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv3d14_out1);
        let conv3d15_out1 = self.conv3d15.forward(relu11_out1);
        let add4_out1 = relu9_out1.add(conv3d15_out1);
        let relu12_out1 = burn::tensor::activation::relu(add4_out1);
        let conv3d16_out1 = self.conv3d16.forward(relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv3d16_out1);
        let conv3d17_out1 = self.conv3d17.forward(relu13_out1.clone());
        let transpose1_out1 = conv3d17_out1.permute([0, 2, 3, 4, 1]);
        let conv3d18_out1 = self.conv3d18.forward(relu13_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv3d18_out1);
        let reshape2_out1 = sigmoid1_out1.reshape([1, 512, 256, 256, 1]);
        (reshape2_out1, transpose1_out1)
    }
}

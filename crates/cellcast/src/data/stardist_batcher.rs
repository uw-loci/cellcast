use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use ndarray::Array2;

/// TODO
#[derive(Clone, Debug)]
pub struct StarDist2DBatch {
    pub test: Tensor<4>,
    pub train: Tensor<4>,
}

impl Batcher<Array2<u16>, StarDist2DBatch> for StarDist2DBatch {
    fn batch(&self, items: Vec<Array2<u16>>, device: &Device) -> StarDist2DBatch {
        todo!();
    }
}

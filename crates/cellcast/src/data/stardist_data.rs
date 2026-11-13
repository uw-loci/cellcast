use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use ndarray::Array2;

/// A StarDist2D training batch.
#[derive(Clone, Debug)]
pub struct StarDist2DBatch {
    pub images: Tensor<4>,
    pub targets: Tensor<4>,
}

/// A StarDist2D training batcher.
///
/// This struct implements the `Batcher` burn trait.
#[derive(Clone, Default)]
pub struct StarDist2DBatcher {}

/// A StarDist2D training batch item.
#[derive(Clone, Debug)]
pub struct StarDist2DItem {
    pub image: Array2<f32>,
    pub label: Array2<u64>,
}

impl Batcher<StarDist2DItem, StarDist2DBatch> for StarDist2DBatcher {
    fn batch(&self, items: Vec<StarDist2DItem>, device: &Device) -> StarDist2DBatch {
        let images = items
            .iter()
            .map(|item| {
                let shape = item.image.shape();
                let (data, _) = item.image.clone().into_raw_vec_and_offset();
                TensorData::new(data, shape)
            })
            .map(|tensor_data| Tensor::<4>::from_data(tensor_data, device))
            .collect::<Vec<Tensor<4>>>();
        let targets = items
            .iter()
            .map(|item| {
                let shape = item.label.shape();
                let (data, _) = item.label.clone().into_raw_vec_and_offset();
                TensorData::new(data, shape)
            })
            .map(|tensor_data| Tensor::<4>::from_data(tensor_data, device))
            .collect::<Vec<Tensor<4>>>();
        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);
        StarDist2DBatch { images, targets }
    }
}

impl StarDist2DItem {
    /// Create a new StarDist2D batch item from an image and label pair.
    pub fn new(image: Array2<f32>, label: Array2<u64>) -> Self {
        Self { image, label }
    }
}

pub fn create_training_batch(
    images: Vec<Array2<f32>>,
    labels: Vec<Array2<u64>>,
) -> StarDist2DBatch {
    // todo -> ensure images len matches labels len.
    // todo -> ensure each image dim matches its paired label dim.
    // todo -> the other things i forgot
    todo!();
}

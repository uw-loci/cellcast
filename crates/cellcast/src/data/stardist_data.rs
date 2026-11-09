use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use imgal::prelude::*;
use imgal::copy::copy_into_flat;
use ndarray::Array2;

/// A StarDist2D training batch.
#[derive(Clone, Debug)]
pub struct StarDist2DBatch {
    pub images: Tensor<4>,
    pub labels: Tensor<4>,
}

/// A StarDist2D training batcher.
///
/// This struct implements the `Batcher` burn trait.
#[derive(Clone, Default)]
pub struct StarDist2DBatcher {}

/// A StarDist2D training batch item.
#[derive(Clone, Debug)]
pub struct StarDist2DItem<T>
where
    T: AsNumeric,
{
    pub image: Array2<T>,
    pub label: Array2<u64>,
}

impl<T> Batcher<StarDist2DItem<T>, StarDist2DBatch> for StarDist2DBatcher
where
    T: AsNumeric,
{
    fn batch(&self, items: Vec<StarDist2DItem<T>>, device: &Device) -> StarDist2DBatch {
        let images = items.iter()
            .map(|item| {
                todo!();
            });
        todo!();
    } 
}

impl<T> StarDist2DItem<T>
where
    T: AsNumeric,
{
    /// Create a new StarDist2D batch item from an image and label pair.
    pub fn new(image: Array2<T>, label: Array2<u64>) -> Self {
        Self { image, label }
    }
}

pub fn create_training_batch<T>(images: Vec<Array2<T>>, labels: Vec<Array2<u64>>) -> StarDist2DBatch {
    // todo -> ensure images len matches labels len.
    // todo -> ensure each image dim matches its paired label dim.
    // todo -> the other things i forgot
    todo!();
}

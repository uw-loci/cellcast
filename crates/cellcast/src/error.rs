use std::error::Error;
use std::fmt;

use imgal::prelude::*;

#[derive(Debug)]
pub enum CellcastError {
    Imgal(ImgalError),
}

impl fmt::Display for CellcastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CellcastError::Imgal(err) => write!(f, "{}", err),
        }
    }
}

impl Error for CellcastError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CellcastError::Imgal(err) => Some(err),
        }
    }
}

impl From<ImgalError> for CellcastError {
    fn from(err: ImgalError) -> Self {
        CellcastError::Imgal(err)
    }
}

//! Labeling functions for 2D and 3D data.
//!
//! This module provides functions for converting 2D and 3D data (*i.e.*
//! polygons and polyhedrons) into label images.

pub(crate) mod polygon_label;
pub(crate) mod polyhedron_label;
pub use polygon_label::distance_polygon_to_label;
pub use polyhedron_label::distance_polyhedron_to_label;

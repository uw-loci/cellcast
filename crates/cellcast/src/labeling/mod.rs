//! Labeling functions for 2D and 3D data.
pub(crate) mod polygon_label;
pub(crate) mod polyhedron_label;
pub use polygon_label::distance_polygon_to_label;
pub use polyhedron_label::distance_polyhedron_to_label;

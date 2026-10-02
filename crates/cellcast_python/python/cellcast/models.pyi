"""
Cell segmentation models.

This module contains the supported cellcast cell segmentation models. Each
model is first initialized on the GPU or CPU with either fetched pre-trained
weights or custom weights.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, final

@final
class StarDist2D:
    """
    A StarDist2D instance segmentation model.

    An initialized StarDist2D instance segmentation model with pre-trained or
    custom weights for 2D fluorescence (`fluo`) or H&E-stained (`he`) images.
    The model runs on either a CPU or GPU backend as determined at
    initialization time.
    """
    @staticmethod
    def init_fluo(
        weights_path: str | None = None, gpu: bool | None = None
    ) -> StarDist2D:
        """
        Initialize a StarDist2D fluo model.

        Initializes a StarDist2D fluo model using the versatile fluo pre-trained
        weights or custom weights. A StarDist2D model can be initialized on either
        the GPU or CPU, but not both concurrently. The model is pre-warmed as part
        of the initializtion process.

        Args:
            weights_path: The path to custom StarDist2D weights in burnpack (`.bpk`)
                format. If `None` then the versatile fluo pre-trained weights are
                used.
            gpu: If `True`, the GPU backend is used. If `False` then the CPU backend
                is used.

        Returns:
            An initialized StarDist2D fluo model.

        Errors:
            If the requested model can not be initialized.
        """
    @staticmethod
    def init_he(weights_path: str | None = None, gpu: bool | None = None) -> StarDist2D:
        """
        Initialize a StarDist2D HE model.

        Initializes a StarDist2D Fluo model using the versatile HE pre-trained
        weights or custom weights. A StarDist2D model can be initialized on either
        the GPU or CPU, but not both concurrently. The model is pre-warmed as part
        of the initializtion process.

        Args:
            weights_path: The path to custom StarDist2D weights in burnpack (`.bpk`)
                format. If `None` then the versatile HE pre-trained weights are used.
            gpu: If `True`, the GPU backend is used. If `False` then the CPU backend
                is used.

        Returns:
            An initialized StarDist2D HE model.

        Errors:
            If the requested model can not be initialized.
        """
    def predict_fluo(
        self,
        /,
        data: Any,
        pmin: float | None = None,
        pmax: float | None = None,
        prob_threshold: float | None = None,
        nms_threshold: float | None = None,
    ) -> Incomplete:
        """
        Predict instance segmentation labels with the StarDist2D fluo model.

        Performs model inference with the StarDist2D fluo model, returning instance
        segmentations of star-convex shapes.

        Args:
            data: The input 2D image.
            pmin: The minimum percentage to linear percentile normalize the input
                image. If `None`, then `pmin = 1.0`.
            pmax: The maximum percentage to linear percentile normalize the input
                image. If `None`, then `pmax = 99.8`.
            prob_threshold: The object/polygon probability threshold. If `None`,
                then `prob_threshold == 0.479071463157368`.
            nms_threshold: The non-maximum suppression (NMS) threshold. If `None`,
                then `nms_threshold == 0.3`.

        Returns:
            The StarDist2D fluo model instance segmentation label image.

        Errors:
            If `pmin` and/or `pmax` are outside of range `0.0` to `1.0.`

        Reference
            <https://doi.org/10.1007/978-3-030-00934-2_30>
        """
    def predict_he(
        self,
        /,
        data: Any,
        pmin: float | None = None,
        pmax: float | None = None,
        prob_threshold: float | None = None,
        nms_threshold: float | None = None,
        axis: int | None = None,
    ) -> Incomplete:
        """
        Predict instance segmentation labels with the StarDist2D HE model.

        Performs model inference with the StarDist2D HE model, returning instance
        segmentations of star-convex shapes.

        Args:
            data: The input 3D image, where the third dimension is the channel axis.
            pmin: The minimum percentage to linear percentile normalize the input
                image. If `None`, then `pmin = 1.0`.
            pmax: The maximum percentage to linear percentile normalize the input
                image. If `None`, then `pmax = 99.8`.
            prob_threshold: The object/polygon probability threshold. If `None`,
                then `prob_threshold == 0.6924782541382084`.
            nms_threshold: The non-maximum suppression (NMS) threshold. If `None`,
                then `nms_threshold == 0.3`.
            axis: The channel axis. If `None` then `axis == 2`.

        Returns:
            The StarDist2D HE model instance segmentation label image.

        Errors:
            If `pmin` and/or `pmax` are outside of range `0.0` to `1.0.`

        Reference
            <https://doi.org/10.1007/978-3-030-00934-2_30>
        """

@final
class StarDist3D:
    """
    A StarDist3D instance segmentation model.

    An initialized StarDist3D instance segmentation model with pre-trained or
    custom weights for 3D volumetric fluorescence (`fluo`) images. The model
    runs on either a CPU or GPU backend as determined at initialization time.
    """
    @staticmethod
    def init_fluo(
        weights_path: str | None = None,
        anisotropy: Sequence[float] | None = None,
        gpu: bool | None = None,
    ) -> StarDist3D:
        """
        Initialize a StarDist3D fluo model.

        Initializes a StarDist3D fluo model using the versatile fluo pre-trained
        weights or custom weights. A StarDist3D model can be initialized on either
        the GPU or CPU, but not both concurrently. The model is pre-warmed with as
        part of the initializtion process.

        Args:
            weights_path: The path to custom StarDist3D weights in burnpack (`.bpk`)
                format. If `None` then the versatile fluo pre-trained weights are
                used.
            anisotropy: The anisotropy the model was trained with for all three
                axes. If `None` then anisotropy of `[2.0, 1.0, 1.0]` is used.
            gpu: If `True`, the configured GPU backend is used. If `false` then the
                configured CPU backend is used.

        Returns:
            An initialized StarDist3D fluo model.

        Errors:
            If the requested model can not be initialized. If
            `anisotropy.len() != 3`.
        """
    def predict_fluo(
        self,
        /,
        data: Any,
        pmin: float | None = None,
        pmax: float | None = None,
        prob_threshold: float | None = None,
        nms_threshold: float | None = None,
        axis: int | None = None,
    ) -> Incomplete:
        """
        Predict instance segmentation labels with the StarDist3D fluo model.

        Performs model inference with the StarDist3D fluo model, returning instance
        segmentations of star-convex shapes.

        Args:
            data: The input 3D image.
            pmin: The minimum percentage to linear percentile normalize the input
                image. If `None`, then `pmin = 1.0`.
            pmax: The maximum percentage to linear percentile normalize the input
                image. If `None`, then `pmax = 99.8`.
            prob_threshold: The object/polyhedron probability threshold. If `None`,
                then `prob_threshold == 0.7079326182611463`.
            nms_threshold: The non-maximum suppression (NMS) threshold. If `None`,
                then `nms_threshold == 0.3`.
            axis: The `pln` or `z` axis. If `None` then `axis == 0`.

        Returns
            The StarDist3D fluo model instance segmentation label image.

        Errors:
            If `pmin` and/or `pmax` are outside of range `0.0` to `1.0.` If
            `axis >= 3`.

        Reference
            <https://doi.org/10.1109/WACV45572.2020.9093435>
        """

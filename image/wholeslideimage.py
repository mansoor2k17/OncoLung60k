import warnings
from pathlib import Path
from typing import List, Tuple, Type, Union

import cv2
import numpy as np

from wholeslidedata.annotation.types import Annotation
from wholeslidedata.image.backend import WholeSlideImageBackend
from wholeslidedata.image.backends import get_backend
from wholeslidedata.image.spacings import (
    calculate_ratios_and_spacings,
    take_closest_level,
)
from wholeslidedata.image.utils import mask_patch_with_annotation


class WholeSlideImage:
    SPACING_MARGIN = 0.3

    def __init__(
        self,
        path: Union[Path, str],
        backend: Union[Type[WholeSlideImageBackend], str] = "openslide",
        auto_resample: bool = False,
    ) -> None:
        """WholeSlideImage that can open en sample from whole slide images

        Args:
            path (Union[Path, str]): path to whole slide image file
            backend (Union[WholeSlideImageBackend, str], optional): image backend that opens and extracts regions from the whole slide image. Defaults to 'openslide'.
        """

        self._path = path
        self._backend = get_backend(backend)(path=self._path)
        self._auto_resample = auto_resample
        self._shapes = self._backend._init_shapes()
        self._downsamplings = self._backend._init_downsamplings()
        self._spacings = self._backend._init_spacings(self._downsamplings)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self._backend.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def spacings(self) -> List[float]:
        return self._spacings

    @property
    def shapes(self) -> List[Tuple[int, int]]:
        return self._shapes

    @property
    def downsamplings(self) -> List[float]:
        return self._downsamplings

    @property
    def level_count(self) -> int:
        return len(self.spacings)

    def get_downsampling_from_level(self, level: int) -> float:
        return self.downsamplings[level]

    def get_level_from_spacing(self, spacing: float) -> int:
        closest_level = take_closest_level(self._spacings, spacing)
        spacing_margin = spacing * WholeSlideImage.SPACING_MARGIN

        if abs(self.spacings[closest_level] - spacing) > spacing_margin:
            warnings.warn(
                f"spacing {spacing} outside margin (0.3%) for {self._spacings}, returning closest spacing: {self._spacings[closest_level]}"
            )

        return closest_level

    def get_real_spacing(self, spacing):
        level = self.get_level_from_spacing(spacing)
        return self._spacings[level]

    def get_downsampling_from_spacing(self, spacing: float) -> float:
        level = self.get_level_from_spacing(spacing)
        return self.get_downsampling_from_level(level)

    def get_shape_from_spacing(self, spacing: float) -> float:
        level = self.get_level_from_spacing(spacing)
        return self.shapes[level]

    def _get_resampled_patch(
        self, x: int, y: int, width: int, height: int, quantize_ratio: float, level: int
    ):
        patch = self._backend.get_patch(
            x, y, int(width * quantize_ratio), int(height * quantize_ratio), level
        )
        return cv2.resize(patch, (width, height))

    def _get_patch_auto_resampled(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        center: bool = True,
        relative: bool = False,
    ):
        (
            coordinate_ratio,
            sample_ratio,
            quantize_ratio,
            quantized_sample_spacing,
        ) = calculate_ratios_and_spacings(self.spacings, spacing, relative)

        level = self.get_level_from_spacing(quantized_sample_spacing)

        x, y = x * coordinate_ratio, y * coordinate_ratio

        if quantize_ratio != 1:
            if center:
                x, y = x - quantize_ratio * (width // 2), y - quantize_ratio * (
                    height // 2
                )
            return self._get_resampled_patch(x, y, width, height, quantize_ratio, level)

        if center:
            x, y = x - sample_ratio * (width // 2), y - sample_ratio * (height // 2)
        return self._backend.get_patch(x, y, width, height, level)

    def _get_patch(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        center: bool = True,
        relative: bool = False,
    ):
        level = self.get_level_from_spacing(spacing)
        if relative is True:
            rel_downsampling = int(self.get_downsampling_from_spacing(spacing))
            x, y = x * rel_downsampling, y * rel_downsampling
        elif type(relative) in (float, int):
            rel_downsampling = int(self.get_downsampling_from_spacing(relative))
            x, y = x * rel_downsampling, y * rel_downsampling

        if center:
            downsampling = int(self.get_downsampling_from_spacing(spacing))
            x, y = x - downsampling * (width // 2), y - downsampling * (height // 2)

        return self._backend.get_patch(x, y, width, height, level)

    def get_patch(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        center: bool = True,
        relative: bool = False,
    ) -> np.ndarray:
        """Extracts a patch/region from the wholeslideimage

        Args:
            x (int): x value
            y (int): y value
            width (int): width of region
            height (int): height of region
            spacing (float): spacing/resolution of the patch
            center (bool, optional): if x,y values are centres or top left coordinated. Defaults to True.
            relative (bool, optional): if x,y values are a reference to the dimensions of the specified spacing. Defaults to False.
            auto_resample (bool, optional): if spacing is not available, auto resample patch to specified spacing. Defaults to False.

        Returns:
            np.ndarray: numpy patch
        """

        if self._auto_resample:
            return self._get_patch_auto_resampled(
                x, y, width, height, spacing, center, relative
            )
        return self._get_patch(x, y, width, height, spacing, center, relative)

    def get_slide(self, spacing):
        if spacing < 2.0:
            warnings.warn(
                f"Trying to load slide at spacing<2.0...",
            )

        level = self.get_level_from_spacing(spacing)
        shape = self.shapes[level]
        return self.get_patch(0, 0, *shape, spacing, center=False)

    def get_region_from_annotations(
        self,
        annotations: List[Annotation],
        spacing: float,
        margin: int = 0,
        masked=False,
    ):
        scaling = self._spacings[0] / self.get_real_spacing(spacing)

        bounds = np.array(
            [[annotation.bounds[i] for i in range(4)] for annotation in annotations]
        )

        min_x = min(bounds[..., 0]) - margin // 2
        min_y = min(bounds[..., 1]) - margin // 2
        max_x = max(bounds[..., 2]) + margin // 2
        max_y = max(bounds[..., 3]) + margin // 2

        width = (max_x - min_x) * scaling
        height = (max_y - min_y) * scaling

        patch = self.get_patch(
            min_x,
            min_y,
            width,
            height,
            spacing=spacing,
            center=False,
        )

        if not masked:
            return patch

        new_patch = np.zeros(shape=patch.shape, dtype="uint8")
        for annotation in annotations:
            new_patch += mask_patch_with_annotation(
                patch, annotation, scaling, offset=(min_x, min_y)
            )

        return new_patch

    def __repr__(self):
        return str(self.path)

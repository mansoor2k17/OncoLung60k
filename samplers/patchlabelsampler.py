from typing import List, Optional

import cv2
import numpy as np
from skimage.transform import resize
from wholeslidedata.annotation.types import PointAnnotation, PolygonAnnotation
from wholeslidedata.samplers.utils import shift_coordinates
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.samplers.sampler import Sampler


class PatchLabelSampler(Sampler):
    """[summary]

    Args:
        Sampler ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    def sample(
        self,
        wsa,
        point: PointAnnotation,
        size,
        ratio,
    ):
        """[summary]

        Args:
            wsa ([type]): [description]
            point ([type]): [description]
            size ([type]): [description]
            ratio ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

    def resolve_batch(y_batch):
        pass


class MaskPatchLabelSampler(PatchLabelSampler):
    def __init__(self, image_backend, ratio, center, relative, spacing=0.5):
        self._image_backend = image_backend
        self._ratio = ratio
        self._center =center
        self._relative = relative
        self._spacing = spacing

    # annotation should be coupled to image_annotation. how?
    def sample(
        self,
        wsa,
        point,
        size,
        ratio,
    ):
        x, y = point
        width, height = size
        mask = WholeSlideImage(wsa.path, backend=self._image_backend)
        spacing = mask.spacings[0]

        mask_patch = np.array(
            mask.get_patch(
                int(x // self._ratio),
                int(y // self._ratio),
                int(width // self._ratio),
                int(height // self._ratio),
                spacing=self._spacing,
                center=self._center,
                relative=self._relative,
            )
        )[..., 0]

        mask.close()
        mask = None
        del mask

        # upscale
        if self._ratio > 1:
            mask_patch = resize(
                mask_patch.squeeze().astype("uint8"),
                size,                
                order=0,
                preserve_range=True,
            )

        return mask_patch.astype(np.uint8)

    def resolve_batch(y_batch):
        pass


class SegmentationPatchLabelSampler(PatchLabelSampler):

    # annotation should be coupled to image_annotation. how?
    def sample(
        self,
        wsa,
        point,
        size,
        ratio,
    ):
        center_x, center_y = point
        width, height = size

        # get annotations
        annotations = wsa.select_annotations(
            center_x, center_y, (width * ratio) - 1, (height * ratio) - 1
        )

        # create mask placeholder
        mask = np.zeros((height, width), dtype=np.int32)
        # set labels of all selected annotations
        for annotation in annotations:
            coordinates = np.copy(annotation.coordinates)
            coordinates = shift_coordinates(
                coordinates, center_x, center_y, width, height, ratio
            )

            if isinstance(annotation, PolygonAnnotation):
                holemask = np.ones((height, width), dtype=np.int32) * -1
                for hole in annotation.holes:
                    hcoordinates = shift_coordinates(
                        hole, center_x, center_y, width, height, ratio
                    )
                    cv2.fillPoly(holemask, np.array([hcoordinates], dtype=np.int32), 1)
                    holemask[holemask != -1] = mask[holemask != -1]
                cv2.fillPoly(
                    mask,
                    np.array([coordinates], dtype=np.int32),
                    annotation.label.value,
                )
                mask[holemask != -1] = holemask[holemask != -1]

            elif isinstance(annotation, PointAnnotation):
                mask[int(coordinates[1]), int(coordinates[0])] = annotation.label.value

        return mask.astype(np.uint8)


class ClassificationPatchLabelSampler(PatchLabelSampler):
    def sample(
        self,
        wsa,
        point,
        size,
        ratio,  
    ):
        center_x, center_y = point

        # get annotations
        annotations = wsa.select_annotations(center_x, center_y, 1, 1)

        return np.array([annotations[-1].label.value])


class DetectionPatchLabelSampler(PatchLabelSampler):
    def __init__(
        self,
        max_number_objects: int = None,
        detection_labels: List[str] = None,
        point_box_sizes: Optional[dict] = None,
    ):
        self._max_number_objects = max_number_objects
        self._point_box_sizes = point_box_sizes
        self._detection_labels = detection_labels

    def sample(
        self,
        wsa,
        point,
        size,
        ratio,
    ):
        center_x, center_y = point
        width, height = size

        # Get annotations
        annotations = wsa.select_annotations(
            center_x, center_y, (width * ratio) - 1, (height * ratio) - 1
        )

        max_number_objects = self._max_number_objects or width * height
        if len(annotations) > max_number_objects:
            raise ValueError(
                f"to many objects in ground truth: {len(annotations)} with possible max number of objects: {max_number_objects}"
            )

        objects = np.zeros((max_number_objects, 6))
        idx = 0
        for annotation in annotations:
            if self._detection_labels is not None:
                if annotation.label.name not in self._detection_labels:
                    continue

            if isinstance(annotation, PointAnnotation):
                box_coords = self._get_point_coordinates(
                    annotation, center_x, center_y, width, height, ratio
                )

            if isinstance(annotation, PolygonAnnotation):
                box_coords = self._get_polygon_coordinates(
                    annotation, center_x, center_y, width, height, ratio
                )
            if box_coords is None:
                continue
                
            objects[idx][:4] = box_coords
            objects[idx][4] = annotation.label.value
            objects[idx][5] = 1  # confidence
            idx += 1
        return objects

    def _get_point_coordinates(
        self, annotation, center_x, center_y, width, height, ratio
    ):
        coordinates = shift_coordinates(
            annotation.coordinates, center_x, center_y, width, height, ratio
        )

        size = 0
        if self._point_box_sizes is not None:
            size = np.array(self._point_box_sizes[annotation.label.name])

        x1 = int(max(0, coordinates[0] - (size // 2)))
        y1 = int(max(0, coordinates[1] - (size // 2)))
        x2 = int(min(width-1, coordinates[0] + (size // 2)))
        y2 = int(min(height-1, coordinates[1] + (size // 2)))
        if x2 <= x1:
            return None
        if y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _get_polygon_coordinates(
        self, annotation, center_x, center_y, width, height, ratio
    ):
        coordinates = [annotation.bounds[:2], annotation.bounds[2:]]
        coordinates = np.array(coordinates, dtype="float")
        coordinates = shift_coordinates(
            coordinates, center_x, center_y, width, height, ratio
        )
        x1 = int(max(0, coordinates[0][0]))
        y1 = int(max(0, coordinates[0][1]))
        x2 = int(min(width-1, coordinates[1][0]))
        y2 = int(min(height-1, coordinates[1][1]))
        if x2 <= x1:
            return None
        if y2 <= y1:
            return None
        return x1, y1, x2, y2

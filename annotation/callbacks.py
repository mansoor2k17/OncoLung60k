
from typing import List

from wholeslidedata.annotation.types import Annotation
from shapely import geometry

class AnnotationCallback:
    def __call__(self, annotations: List[Annotation]):
        return annotations

class ScalingAnnotationCallback(AnnotationCallback):

    def __init__(self, scaling):
        self._scaling = scaling

    def __call__(self, annotations: List[Annotation]):
        scaled_annotations = []
        for annotation in annotations:
            scaled_annotation = {}
            scaled_annotation["index"] = annotation.index
            scaled_annotation["coordinates"] = annotation.coordinates * self._scaling
            scaled_annotation["label"] = annotation.label.todict()
            scaled_annotations.append(Annotation.create(**scaled_annotation))
        return scaled_annotations

class TiledAnnotationCallback(AnnotationCallback):

    def __init__(self, tile_size, label_names, ratio=1, overlap=0, full_coverage=False):
        self._tile_size = tile_size*ratio
        self._overlap = overlap*ratio
        self._full_coverage = full_coverage
        self._label_names= label_names

    def __call__(self, annotations: List[Annotation]):
        new_annotations = []
        index=0
        for annotation in annotations:
            if annotation.label.name not in self._label_names:
                annotation._index = index
                new_annotations.append(annotation)
                index += 1
                continue
                
            x1, y1, x2,y2 = annotation.bounds
            for x in range(x1, x2, self._tile_size-self._overlap):
                for y in range(y1, y2, self._tile_size-self._overlap):
                    box_poly = geometry.box(x, y, x+self._tile_size, y+self._tile_size)
                    if not self._full_coverage or box_poly.within(annotation):
                        new_annotations.append(Annotation.create(
                            index=index,
                            coordinates=box_poly.exterior.coords,
                            label=annotation.label.todict(),
                                ))
                        index += 1

        return new_annotations
from typing import Callable, Dict, Iterator
import abc
import numpy as np
from wholeslidedata.samplers.sampler import Sampler
from wholeslidedata.data.dataset import WholeSlideDataSet

class AnnotationSampler(Sampler, Iterator):
    def __init__(self, counts_per_label: Dict, seed: int):
        super().__init__(seed=seed)
        self._counts_per_label = dict(sorted(counts_per_label.items()))

    def __next__(self) -> Callable:
        return self._next

    @abc.abstractmethod
    def _next(self, label: str) -> int:
        pass

    @abc.abstractmethod
    def update(self, data):
        pass


class OrderedAnnotationSampler(AnnotationSampler):
    def __init__(self, counts_per_label, seed):
        super().__init__(counts_per_label=counts_per_label, seed=seed)
        self._counters = {label: 0 for label in self._counts_per_label.keys()}
        self.reset()

    def _next(self, label):
        annotation_index = self._counters[label]
        self._counters[label] += 1
        if self._counters[label] == self._counts_per_label[label]:
            self._reset_label(label)
        return annotation_index

    def _reset_label(self, label):
        self._counters[label] = 0

    def update(self, data):
        pass

    def reset(self):
        self.set_seed()
        for label in self._counts_per_label.keys():
            self._reset_label(label)


class BalancedAnnotationSampler(AnnotationSampler):
    def __init__(self, counts_per_label, seed, random_reset=False):
        super().__init__(counts_per_label, seed=seed)
        self._counters = {
            label: self._random_index_iterator(label)
            for label in self._counts_per_label
        }
        self._random_reset = random_reset

    def _next(self, label):
        try:
            return next(self._counters[label])
        except StopIteration:
            self._reset_label(label)
            return next(self._counters[label])

    def _random_index_iterator(self, label):
        return iter(self._rng.permutation(range(self._counts_per_label[label])))

    def update(self, data):
        pass

    def _reset_label(self, label):
        self._counters[label] = self._random_index_iterator(label)

    def reset(self):
        self.set_seed(reseed=self._random_reset)
        for label in self._counts_per_label.keys():
            self._reset_label(label)


class AreaAnnotationSampler(AnnotationSampler):
    def __init__(self, counts_per_label: dict, seed, dataset: WholeSlideDataSet, weight: float=1.0):
        super().__init__(counts_per_label, seed=seed)
        self._weight = weight
        self._area_annotation_map = {label: {} for label in counts_per_label}
        self._total_area = {label: 0 for label in counts_per_label}

        self._area_annotations = {
            label: np.zeros(counts) for label, counts in counts_per_label.items()
        }

        for label, sample_references in dataset.sample_references.items():
            for annotation_index, sample_reference in enumerate(sample_references):
                annotation = dataset.get_annotation_from_reference(sample_reference)
                self._area_annotation_map[label][self._total_area[label]] = annotation_index
                self._area_annotations[label][annotation_index] = self._total_area[label]
                self._total_area[label] += annotation.area ** self._weight
        self.reset()

    def _next(self, label):
        rint = self._rng.randint(self._total_area[label])
        area_annotation = np.where((rint >= self._area_annotations[label]))[0][-1]
        return self._area_annotation_map[label][
            self._area_annotations[label][area_annotation]
        ]

    def update(self, data):
        pass

    def _reset_label(self, label):
        pass

    def reset(self):
        super().set_seed()

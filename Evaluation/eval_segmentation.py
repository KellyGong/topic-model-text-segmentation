"""
Adapted from https://github.com/sebastianarnold/SECTOR

"""
import numpy as np


class SegmentationMetric:
    def __init__(self, k):
        self.k = k
        self.countDocs = 0
        self.countExp = 0
        self.countPred = 0
        self.pksum = 0

    def update(self, reference:np.array, hypothesis:np.array):
        """
        :param reference: ground truth of segmentation, 1 represent begin of segment, otherwise 0
        :param hypothesis: predict result of segmentation, expression same as reference
        :return: Pk value
        """
        assert len(reference) == len(hypothesis)
        self.pksum += self._calculatePk(hypothesis, reference)
        self.countExp += len(self.get_masses_array(reference))
        self.countPred += len(self.get_masses_array(hypothesis))
        self.countDocs += 1

    def get_pk(self):
        if self.countDocs == 0:
            return 0
        return self.pksum / self.countDocs

    def _calculatePk(self, hypothesis, reference):
        position_reference = self.get_position_array(reference)
        position_hypothesis = self.get_position_array(hypothesis)
        sum = 0
        count = 0
        for t in range(len(position_reference) - self.k):
            agreeRef = True if position_reference[t] == position_reference[t + self.k] else False
            agreeHyp = True if position_hypothesis[t] == position_hypothesis[t + self.k] else False
            if agreeRef != agreeHyp:
                sum += 1
            count += 1
        if len(reference) == 2:
            assert count == 0
            agreeRef = reference[0] == reference[1]
            agreeHyp = hypothesis[0] == hypothesis[1]
            if agreeRef == agreeHyp:
                return 0
            else:
                return 1
        if len(reference) == 1:
            return 0
        return sum / count if count > 0 else 0

    @staticmethod
    def get_position_array(binary_array: np.array):
        """
        :param binary_array: [1, 0, 0, 1, 0, 0, 1, 0]
        :return: [1, 1, 1, 2, 2, 2, 3, 3]
        """
        return np.cumsum(binary_array)

    @staticmethod
    def get_masses_array(binary_array: np.array):
        """
        :param binary_array: [1, 0, 0, 1, 0, 0, 1, 0]
        :return: [3, 3, 2]
        """
        binary_array = np.append(binary_array, 1)
        masses_array = np.where(binary_array == 1)[0]
        for i in range(len(masses_array) - 1, 0, -1):
            masses_array[i] = masses_array[i] - masses_array[i - 1]
        return np.delete(masses_array, 0)








"""
Adapted from https://github.com/sebastianarnold/SECTOR

"""
import numpy as np


class TopicClassificationMetric:
    def __init__(self, k=3):
        self.k = k
        self.countDocs = 0
        self.countExamples = 0
        self.mrrsum = 0
        self.mapsum = 0
        self.p1sum = 0
        self.r1sum = 0
        self.pksum = 0
        self.rksum = 0

    def update(self, Y:np.array, Z:np.array):
        Zi = np.argsort(-Z)
        self.mapsum += self.AP(Y, Z, Zi)
        self.mrrsum += self.RR(Y, Z, Zi)
        self.p1sum += self.Prec(Y, Z, Zi, 1)
        self.r1sum += self.Rec(Y, Z, Zi, 1)
        self.pksum += self.Prec(Y, Z, Zi, self.k)
        self.rksum += self.Rec(Y, Z, Zi, self.k)
        self.countExamples += 1

    def update_docs(self):
        self.countDocs += 1

    def get_mrr(self):
        return self.mrrsum / self.countExamples

    def get_map(self):
        return self.mapsum / self.countExamples

    def get_recallK(self):
        return self.rksum / self.countExamples

    def get_precisionK(self):
        return self.pksum / self.countExamples

    def get_recall1(self):
        return self.r1sum / self.countExamples

    def get_precision1(self):
        return self.p1sum / self.countExamples

    @staticmethod
    def AP(Y, Z, Zi):
        sum = 0
        count = 0
        for k in range(len(Y)):
            idx = Zi[k]
            if Y[idx] > 0:
                sum += TopicClassificationMetric.Prec(Y, Z, Zi, k+1)
                count += 1

        assert count == 1
        if count > 0:
            return sum / count
        else:
            return 0

    @staticmethod
    def RR(Y, Z, Zi):
        # Mean_reciprocal_rank
        ri = np.argmax(Y)
        if ri >= 0:
            r = TopicClassificationMetric.rank(ri, Zi)
            return 1 / r
        else:
            return 0

    @staticmethod
    def rank(ri, Zi):
        for i in range(len(Zi)):
            if Zi[i] == ri:
                return i + 1
        raise ValueError

    @staticmethod
    def Prec(Y, Z, Zi, k):
        sum = 0
        for i in range(k):
            idx = Zi[i]
            if Y[idx] > 0:
                sum += 1

        return sum / k

    @staticmethod
    def Rec(Y, Z, Zi, k):
        sum = 0
        for i in range(k):
            idx = Zi[i]
            if Y[idx] > 0:
                sum += 1

        return sum / np.sum(Y)



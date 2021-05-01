import abc
from typing import List


class PreProcessor(abc.ABC):

    def process_all(self, data: List[List[str]]):
        res = []
        for s in data:
            res.append(self.process(s))
        return res

    @abc.abstractmethod
    def process(self, sent):
        pass


class LowerProcess(PreProcessor):

    def process(self, sent):
        return [s.lower().strip() for s in sent]
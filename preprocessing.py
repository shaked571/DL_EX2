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

    def process(self, w:str):
        return w.lower()


class TitleProcess(PreProcessor):

    def process(self, w:str):
        if w.islower():
            return w
        return w.title()

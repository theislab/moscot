from abc import ABCMeta


class AnalysisMixinMeta(ABCMeta):
    pass


class AnalysisMixin(metaclass=AnalysisMixinMeta):
    def push_forward(self):
        pass

    def pull_backward(self):
        pass

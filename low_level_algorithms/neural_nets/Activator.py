from abc import ABC, abstractmethod
import math

class Activator(ABC):

    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

class Identity(Activator):

    def function(self, x):
        return x

    def derivative(self, x):
        return 1

class Sigmoid(Activator):

    def function(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative(self, x):
        return self.function(x) * (1 - self.function(x))

class ReLu(Activator):

    def function(self, x):
        return max(0, x)

    def derivative(self, x):
        if x <= 0:
            return 0
        else:
            return 1
from abc import ABC

class Optimizador(ABC):
    def __init__(self,funcion:callable):
        self.funcion = funcion
        super().__init__()    
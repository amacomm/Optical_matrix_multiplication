import torch as _torch
import torch.nn as _nn
from .config import Config as _Config
from .propagator import Propagator as _Propagator

class OpticalMul(_nn.Module):
    """
    Класс системы, выполняющей оптически операцию умножения матрицы на вектор.
 
    Поля:
        Lens1: цилиндрическая линза.
        Lens2: скрещенная линза.
    """
    def __init__(self, propagator: _Propagator,
                 config: _Config):
        """
        Конструктор класса.
 
        Параметры:
            propagator: класс распространения светового поля в свободном пространстве.
            config: конфигурация расчётной системы.
        """
        super(OpticalMul, self).__init__()
        self.__propagator: _Propagator = propagator
        self.__config: _Config = config
        self.Lens1: _torch.Tensor = _torch.exp(-1j * config.K / config.distance * config.X**2).chalf()
        self.Lens2: _torch.Tensor = _torch.exp(-1j * config.K / 2 / config.distance * config.Y**2) * self.Lens1
        self.Lens1 = self.Lens1.chalf()
        self.Lens2 = self.Lens2.chalf()
        self.__zero_count = int((config.array_size - config.vector_size) / 2)
        self.__pad_matrix = _nn.ZeroPad2d(self.__zero_count)
        self.__pad_vector = _nn.ZeroPad2d((self.__zero_count, self.__zero_count, int(config.array_size / 2), int(config.array_size / 2) - 1))

    def prepare_vector(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки вектора столбца к подаче на вход системе.

        Параметры:
            data: вектор амплитудного распределения
            (ограничение значений в диапазоне [0, 1]).

        Returns:
            Матрица содержащая входной вектор.
        """
        data = data.half().flip(-1)
        data = data.unsqueeze(-2)
        data = _nn.functional.interpolate(data.to(self.__config.device),
                                          [1, self.__config.vector_size],
                                          mode="nearest")
        return self.__pad_vector(data)

    def prepare_matrix(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки матрицы к подаче на вход системе.

        Параметры:
            data: матрица амплитудного распределения
            (ограничение значений в диапазоне [0, 1]).

        Returns:
            Матрица дополненая до размеров системы.
        """
        self.__avg_pool = _nn.AvgPool1d(int(self.__config.vector_size / data.size(-2)))
        data = _nn.functional.interpolate(data.half().to(self.__config.device),
                                          [self.__config.vector_size, self.__config.vector_size],
                                          mode="nearest")
        data = self.__pad_matrix(data)
        return data

    def prepare_out(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения результата матричного умножения:
        вектора столбца.

        Параметры:
            data: матрица выходого комплексного поля системы.

        Returns:
            Вектор столбец (амплитудное распределение).
        """
        field = field.abs()
        field = field[..., self.__zero_count:self.__config.array_size - self.__zero_count, int(self.__config.array_size / 2)]
        field = self.__avg_pool(field)
        return field.flip(-1)

    def __call__(self, matrix: _torch.Tensor,
                 vector: _torch.Tensor) -> _torch.Tensor:
        """
        Метод умножения матрицы на вектор.

        Параметры:
            matrix: матрица (N, C, H, W).
                WARNING: H <= W.
            vector: вектор столбец (N, C, L).

        Returns:
            Вектор столбец (N, C, H).

        Пример:
            >>> mul = OpticalMul(...)
            >>> matrix = torch.rand((1, 1, 256, 256)) > 0.5
            >>> vector = torch.rand((1, 1, 256)) > 0.5
            >>> mul(matrix, vector).shape
            torch.Size([1, 1, 256])
            >>> matrix = torch.rand((1, 1, 256, 128)) > 0.5
            >>> vector = torch.rand((1, 1, 256)) > 0.5
            >>> mul(matrix, vector).shape
            torch.Size([1, 1, 128])
        """
        vec_field = self.prepare_vector(vector)
        mat_field = self.prepare_matrix(matrix)

        vec_field = self.__propagator(vec_field)
        vec_field = self.__propagator(vec_field * self.Lens2.to(self.__config.device))
        vec_field = self.__propagator(vec_field * self.Lens1.to(self.__config.device) * mat_field)
        vec_field = self.__propagator(vec_field * self.Lens2.chalf().to(self.__config.device).T)

        return self.prepare_out(vec_field)
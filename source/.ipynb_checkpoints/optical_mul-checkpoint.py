import torch as _torch
import torch.nn as _nn
from .config import Config as _Config
from .propagator import Propagator as _Propagator

class OpticalMul(_nn.Module):
    """
    Класс системы, выполняющей оптически операцию умножения матрицы на матрицу.
 
    Поля:
        Lens1: цилиндрическая линза.
        Lens2: скрещенная линза.
    """
    def __init__(self, propagator: _Propagator,
                 config: _Config):
        """
        Конструктор класса.
 
        Args:
            propagator: класс распространения светового поля в свободном пространстве.
            config: конфигурация расчётной системы.
        """
        super(OpticalMul, self).__init__()
        self.propagator: _Propagator = propagator
        self.__config: _Config = config
        
        Lens1 = _torch.exp(-1j * config.K / config.distance * config.X**2)
        Lens2 = _torch.exp(-1j * config.K / 2 / config.distance * config.Y**2) * Lens1
        
        self.register_buffer('Lens1', _torch.view_as_real(Lens1).half(), persistent=True)
        self.register_buffer('Lens2', _torch.view_as_real(Lens2).half(), persistent=True)
        
        self.__zero_count_by_x = int((config.array_size - config.vector_size * config.scale_by_x) / 2)
        
        self.__pad_vector = _nn.ZeroPad2d((self.__zero_count_by_x,
                                           self.__zero_count_by_x,
                                           int(config.array_size / 2 - config.scale_by_y / 2 + 0.5),
                                           int(config.array_size / 2 - config.scale_by_y / 2)))

        kron_vec_utils = _torch.ones((self.__config.scale_by_y, self.__config.scale_by_x))
        kron_mat_utils = _torch.ones((self.__config.scale_by_x, self.__config.scale_by_x))
        self.register_buffer('kron_vec_utils', kron_vec_utils.half(), persistent=True)
        self.register_buffer('kron_mat_utils', kron_mat_utils.half(), persistent=True)

    def prepare_vector(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки матрицы A, как набора векторов столбцов, к подаче на вход системы.

        Args:
            data: матрица амплитудного распределения
            (ограничение значений в диапазоне [0, 1]).

        Returns:
            Матрицы содержащие вектора входной матрицы A.
        """
        data = data.half().flip(-1)
        data = data.unsqueeze(-2)
        data = _torch.kron(data.contiguous(), self.kron_vec_utils)
        return self.__pad_vector(data)

    def prepare_matrix(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки матрицы B к подаче на вход системы.

        Args:
            data: матрица амплитудного/фазового распределения
            (ограничение значений в диапазоне [0, 1] для аплитудного рапределения
            или ограничение значений 1 по амплитуде и [-pi, pi] по фазе).

        Returns:
            Матрица B дополненая до размеров системы.
        """
        if (data.dim() > 4) and data.size(-1) == 2:
            data = _torch.view_as_complex(data)

        data = data.transpose(-2, -1)
        
        self.__avg_pool = _nn.AvgPool2d((1,
                                         int(self.__config.vector_size * self.__config.scale_by_x / data.size(-1))))
        data = _torch.kron(data.contiguous(), self.kron_mat_utils)
        
        self.__zero_count_by_y = int((self.__config.array_size - data.size(-2)) / 2)
        self.__pad_matrix = _nn.ZeroPad2d((self.__zero_count_by_x,
                                           self.__zero_count_by_x,
                                           self.__zero_count_by_y,
                                           self.__zero_count_by_y))
        return self.__pad_matrix(data)

    def prepare_out(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения результата матричного умножения.

        Args:
            data: матрицы выходого распределения комплексного поля системы.

        Returns:
            Вектор столбец (амплитудное распределение).
        """
        field = field.abs()
        field = field[..., self.__zero_count_by_y: -self.__zero_count_by_y, int(self.__config.array_size / 2)]
        field = self.__avg_pool(field)
        return field.flip(-1)

    def forward(self,
                input: _torch.Tensor,
                other: _torch.Tensor) -> _torch.Tensor:
        """
        Метод выполения матричного умножения.

        Args:
            input: матрица (B, C, H, W).
            other: матрица (B, C, W, K).

        Returns:
            Рензультат матричного умножения (B, C, H, K).

        Example:
            >>> mul = OpticalMul(...)
            >>> A = torch.rand((1, 1, 256, 256)) > 0.5
            >>> B = torch.rand((1, 1, 256, 256)) > 0.5
            >>> mul(A, B).shape
            torch.Size([1, 1, 256, 256])
            >>> A = torch.rand((1, 1, 64, 256)) > 0.5
            >>> B = torch.rand((1, 1, 256, 128)) > 0.5
            >>> mul(A, B).shape
            torch.Size([1, 1, 64, 128])
        """
        vec_field = self.prepare_vector(input)
        mat_field = self.prepare_matrix(other)
        
        Lens1 = _torch.view_as_complex(self.Lens1)
        Lens2 = _torch.view_as_complex(self.Lens2)

        vec_field = self.propagator(vec_field)
        vec_field = self.propagator(vec_field * Lens2)
        vec_field = self.propagator(vec_field * Lens1 * mat_field)
        vec_field = self.propagator(vec_field * Lens2.T)

        return self.prepare_out(vec_field)
import torch as _torch
import torch.nn as _nn
from .config import Config as _Config
from .propagator import PropagatorCrossLens as _PropCrossLens, PropagatorСylindLens as _PropСylindLens, PropagatorSinc as _PropSinc, Propagator as _Prop

class OpticalMul(_nn.Module):
    """
    Класс системы, выполняющей оптически операцию умножения матрицы на матрицу.
    """
    def __init__(self, config: _Config):
        """
        Конструктор класса.
 
        Args:
            config: конфигурация расчётной системы.
        """
        super(OpticalMul, self).__init__()

        prop_one = _PropSinc(config.input_vector_plane, config.first_lens_plane, config)
        prop_two = _PropCrossLens(config.first_lens_plane, config)
        prop_three = _PropSinc(config.first_lens_plane, config.matrix_plane, config)
        prop_four = _PropСylindLens(config.matrix_plane, config)
        prop_five = _PropSinc(config.matrix_plane, config.second_lens_plane, config)
        prop_six = _PropCrossLens(config.second_lens_plane, config).T
        prop_seven = _PropSinc(config.second_lens_plane, config.output_vector_plane, config)

        self._propagator_one: _Prop = prop_one + prop_two + prop_three + prop_four
        self._propagator_two: _Prop = prop_five + prop_six + prop_seven

        kron_vec_utils = _torch.ones((config.input_vector_split_y, config.input_vector_split_x))
        kron_mat_utils = _torch.ones((config.matrix_split_x, config.matrix_split_y))
        self.register_buffer('_kron_vec_utils', kron_vec_utils, persistent=True)
        self.register_buffer('_kron_mat_utils', kron_mat_utils, persistent=True)
        
        self._avg_pool = _nn.AvgPool2d((1, config.result_vector_split))

    def prepare_vector(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки матрицы левой матрицы, как набора векторов столбцов, к подаче на вход системы.

        Args:
            data: матрица комплексной амплитуды распределений световых полей.

        Returns:
            Матрицы содержащие вектора левой матрицы.
        """
        data = data.cfloat().flip(-1)
        data = data.unsqueeze(-2)
        data = _torch.kron(data.contiguous(), self._kron_vec_utils)
        return data

    def prepare_matrix(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки правой матрицы к подаче на вход системы.

        Args:
            data: матрица комплексной амплитуды распределения светового поля.

        Returns:
            Матрица - оптический элемент в центре модели.
        """
        if (data.dim() > 4) and data.size(-1) == 2:
            data = _torch.view_as_complex(data)

        data = data.cfloat().transpose(-2, -1)
        data = data.unsqueeze(-3)
        data = _torch.kron(data.contiguous(), self._kron_mat_utils)
        return data

    def prepare_out(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения результата матричного умножения.

        Args:
            data: матрицы выходого распределения светового поля системы.

        Returns:
            Вектор столбец (амплитудное распределение).
        """
        ### Закоментированная часть кода - более физически корректный вариант работы модели,
        ### однако, данный вариант кода будет требовать большое кол-во памяти во время обучения
        field = field.abs().squeeze(-1) #**2
        field = self._avg_pool(field)
        return field.flip(-1) #**0.5

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

        vec_field = self._propagator_one(vec_field)
        vec_field = self._propagator_two(vec_field * mat_field)

        return self.prepare_out(vec_field)
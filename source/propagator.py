import torch as _torch
import torch.nn as _nn
import numpy as _np
from scipy.special import fresnel as _fresnel
from .config import ConfigOpticBase as _ConfigOpticBase, ConfigDesignPlane as _ConfigDesignPlane
from typing import Tuple as _Tuple, Sequence as _Sequence

from abc import ABC as _ABC
import collections as _collections

class Propagator(_ABC, _nn.Module):
    """
    Абстрактный класс вычисления распространения светового поля в среде.

    Поля:
        operator_X: оператор отображающий распроcтранение светового поля вдоль оси абсцисс
        operator_Y: оператор отображающий распроcтранение светового поля вдоль оси ординат
    """
    def __init__(self, operator_X: _torch.Tensor, operator_Y: _torch.Tensor):
        super(Propagator, self).__init__()
        operator_X: _torch.Tensor = _torch.view_as_real(operator_X)
        operator_Y: _torch.Tensor = _torch.view_as_real(operator_Y)
        self.register_buffer('_operator_X', operator_X, persistent=True)
        self.register_buffer('_operator_Y', operator_Y, persistent=True)

    @property
    def operator_X(self) -> _torch.Tensor:
        """
        Returns:
            оператор отображающий распроcтранение светового поля вдоль оси абсцисс
        """
        return _torch.view_as_complex(self._operator_X)
    @property
    def operator_Y(self) -> _torch.Tensor:
        """
        Returns:
            оператор отображающий распроcтранение светового поля вдоль оси ординат
        """
        return _torch.view_as_complex(self._operator_Y)

    def __operator_multiplication(self, first_X: _torch.Tensor,
                                  second_X: _torch.Tensor,
                                  first_Y: _torch.Tensor,
                                  second_Y: _torch.Tensor)-> _Tuple[_torch.Tensor, _torch.Tensor]:
        operator_Y = second_Y @ first_Y
        operator_X = first_X @ second_X
        return operator_X, operator_Y

    def cat(self, propagators: _Sequence['Propagator']) -> 'Propagator':
        """
        Метод схлопывания операторов распространения.
    
        Args:
            propagators: последовательность для схлопывания

        Returns:
            новый пропогатор, заменяющих собой серию предыдущих

        Warning:
            порядок расположения пропагаторов в последовательности важен,
            идёт от первого к последниму
        """
        operator_X: _torch.Tensor
        operator_Y: _torch.Tensor
        if not isinstance(propagators, _collections.abc.Sequence):
            operator_X, operator_Y = self.__operator_multiplication(self.operator_X,
                                                               propagators.operator_X,
                                                               self.operator_Y,
                                                               propagators.operator_Y)
        else:
            size = len(propagators)
            operator_X = self.operator_X
            operator_Y = self.operator_Y
            for i in range(size):
                operator_X, operator_Y = self.__operator_multiplication(operator_X,
                                                                   propagators[i].operator_X,
                                                                   operator_Y,
                                                                   propagators[i].operator_Y)
        return Propagator(operator_X, operator_Y)

    def __add__(self, propagator: 'Propagator') -> 'Propagator':
        """
        Метод схлопывания двух пропагаторов.
        Args:
            propagator: пропагатор с которым нужно произвести схлопывание

        Returns:
            новый пропогатор, заменяющих собой оба предыдущих

        Warning:
            операция не комутативная
        """
        return self.cat(propagator)

    def forward(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод распространения светового поля в среде.
 
        Args:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после распространения.
        """
        return self.operator_Y @ field @ self.operator_X

class PropagatorLens(Propagator):
    """
    Абстрактный класс распространения света в тонком оптическом элементе.
    """
    def transpose(self) -> 'PropagatorLens':
        """
        Метод транспонирования тонкого оптического элемента.
        Returns:
           Новый элемент, транспонированный относительно оригинального.
        """
        obj = Propagator.__new__(PropagatorLens)
        Propagator.__init__(obj, self.operator_Y, self.operator_X)
        return obj

    @property
    def T(self) -> 'PropagatorLens':
        """
        Returns:
           Новый элемент, транспонированный относительно текущего.
        """
        return self.transpose()

class PropagatorCrossLens(PropagatorLens):
    """
    Класс распространения света в скрещенной линзе,
    представленной тонким оптическим элементом.
    """
    def __init__(self, plane: _ConfigDesignPlane,
                 config: _ConfigOpticBase):
        """
        Конструктор класса скрещенной линзы.

        Args:
            plane: данные о расчётной плоскости элемента.
            config: данные о световом поле модели.
        """
        operator_X = _torch.exp(-1j * config.K / config.distance * plane.linspace_by_x**2)
        operator_Y = _torch.exp(-1j * config.K / 2 / config.distance * plane.linspace_by_y**2)
        super(PropagatorCrossLens, self).__init__(_torch.diag_embed(operator_X),
                                                  _torch.diag_embed(operator_Y))

class PropagatorСylindLens(PropagatorLens):
    """
    Класс распространения света в цилиндрической линзе,
    представленной тонким оптическим элементом.
    """
    def __init__(self, plane: _ConfigDesignPlane,
                 config: _ConfigOpticBase):
        """
        Конструктор класса цилиндрической линзы.

        Args:
            plane: данные о расчётной плоскости элемента.
            config: данные о световом поле модели.
        """
        operator_X = _torch.exp(-1j * config.K / config.distance * plane.linspace_by_x**2)
        operator_Y = _torch.ones_like(plane.linspace_by_y, dtype=_torch.cfloat)
        super(PropagatorСylindLens, self).__init__(_torch.diag_embed(operator_X),
                                                   _torch.diag_embed(operator_Y))

class PropagatorSinc(Propagator):
    """
    Класс распространения света свободном пространстве
    с использованием разложения по базисным sinc функциям.
    """
    def __init__(self, first_plane: _ConfigDesignPlane,
                 second_plane: _ConfigDesignPlane,
                 config: _ConfigOpticBase):
        """
        Конструктор класса распространения в свободном пространстве.

        Args:
            first_plane: данные о начальной расчётной плоскости.
            second_plane: данные о конечной расчётной плоскости.
            config: данные о световом поле модели.
        """
        operator_X, operator_Y = self.__get_operators(first_plane,
                                                    second_plane,
                                                    config)
        super(PropagatorSinc, self).__init__(operator_X, operator_Y)

    def __get_operator_for_dim(self,
                             pixel_size_in: float,
                             pixel_size_out: float,
                             difference: float,
                             config: _ConfigOpticBase) -> _torch.Tensor:
        bndW = 0.5 / pixel_size_in
        eikz = (_np.exp(1j * config.K * config.distance)**0.5)
        sq2p = (2 / _np.pi)**0.5
        sqzk = ((2 * config.distance / config.K)**0.5)
        mu1 = -_np.pi * sqzk * bndW - difference / sqzk
        mu2 = _np.pi * sqzk * bndW - difference / sqzk
        S1, C1 = _fresnel(mu1 * sq2p)
        S2, C2 = _fresnel(mu2 * sq2p)
        return (((pixel_size_in * pixel_size_out)**0.5 / _np.pi) / sqzk * eikz
                  * _np.exp(0.5j * difference**2 * config.K / config.distance)
                  * (C2 - C1 - 1j * (S2 - S1)) / sq2p)
        
    def __get_operators(self,
                      first_plane: _ConfigDesignPlane,
                      second_plane: _ConfigDesignPlane,
                      config: _ConfigOpticBase) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        difference_x = first_plane.linspace_by_x[None, :] - second_plane.linspace_by_x[:, None]
        difference_y = first_plane.linspace_by_y[None, :] - second_plane.linspace_by_y[:, None]
        operator_X = self.__get_operator_for_dim(first_plane.pixel_size_by_x,
                                               second_plane.pixel_size_by_x,
                                               difference_x,
                                               config).transpose(-2, -1)
        operator_Y = self.__get_operator_for_dim(first_plane.pixel_size_by_y,
                                               second_plane.pixel_size_by_y,
                                               difference_y,
                                               config)
        return operator_X, operator_Y

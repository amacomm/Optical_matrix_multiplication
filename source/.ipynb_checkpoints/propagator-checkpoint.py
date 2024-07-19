import torch as _torch
import torch.nn as _nn
import torch.fft as _fft
import numpy as _np
from scipy.special import fresnel as _fresnel
from scipy.linalg import toeplitz as _toeplitz
from .config import Config as _Config

class Propagator(_nn.Module):
    """
    Класс оператора распространения светового поля в свободном пространстве.
 
    Поля:
        selection: список номеров используемых длин волн.
    """
    def __init__(self, operator,
                 config: _Config = _Config()):
        """
        Конструктор класса.
 
        Параметры:
            operator: оператор распространения светового поля.
            config: конфигурация расчётной системы.
        """
        super().__init__()
        self.set_config(config, operator)
        self.selection = _torch.arange(self.__config.wavelength.size(0))

    def set_config(self, config: _Config, operator):
        """
        Метод замены конфигурационных данных.
 
        Параметры:
            config: конфигурация расчётной системы.
            operator: оператор распространения светового поля.
        """
        self.__config: _Config = config
        self.set_operator(operator)

    def set_operator(self, operator):
        """
        Метод замены оператора распространения светового поля.
 
        Параметры:
            operator: оператор распространения светового поля.
        """
        self.__operator = operator
        self.__operator.set_config(self.__config)

    def get_distance(self) -> float:
        """
        Метод получения дистанции распространения светового поля.
        
        Returns:
            Дистанция распространения светового поля.
        """
        return self.__config.distance

    def propagation(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод распространения светового поля.
 
        Параметры:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после распространения.
        """
        return self.__operator(field, self.__config, self.selection)

    def forward(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод распространения светового поля.
 
        Параметры:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после распространения.
        """
        return self.propagation(field)

class Propagator_ASM_operator:
    """
    Класс оператора разложения светового поля по базису плоских волн.
    """
    def set_config(self, config: _Config):
        """
        Метод замены конфигурационных данных.
 
        Параметры:
            config: конфигурация расчётной системы.
        """
        border = _np.pi * config.array_size / config.aperture_size
        arr = _torch.linspace(-border, border, config.array_size + 1)[:config.array_size]
        xv, yv = _torch.meshgrid(arr, arr, indexing='ij')
        xx = xv**2 + yv**2
        U = _torch.roll(xx, (int(config.array_size / 2), int(config.array_size / 2)), dims = (0, 1))
        p = (config.K[:, None, None].cfloat()**2 - U[None, ...].cfloat())**0.5
        self.__operator = _torch.exp(1j * config.distance * p[:, None, ...])
        
    def __call__(self,
                 field: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод распространения светового поля.
 
        Параметры:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после распространения.
        """
        return _fourier_propagation(self.__operator[selection].to(config.device),
                                     field.to(config.device))

class Propagator_sphere_operator:
    """
    Класс оператора разложения светового поля по базису сферических волн.
    """
    def set_config(self, config: _Config):
        """
        Метод замены конфигурационных данных.
 
        Параметры:
            config: конфигурация расчётной системы.
        """
        r = ((config.X**2 + config.Y**2 + config.distance**2)**0.5)[None, :, :]

        H = ((config.distance / 2 / _np.pi / r**2) * (1 / r - 1j * config.K[:, None, None].cdouble())
             * _torch.exp(1j * config.K[:, None, None].cdouble() * r)).cfloat()
        H = H * config.pixel_size**2
        self.__operator = _fft.fft2(_torch.roll(H, (int(config.array_size/2), int(config.array_size/2)), (1,2)))[:, None, ...]
        
    def __call__(self,
                 field: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод распространения светового поля.
 
        Параметры:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после распространения.
        """
        return _fourier_propagation(self.__operator[selection].to(config.device),
                                     field.to(config.device))

class Propagator_sinc_operator:
    """
    Класс оператора разложения светового поля по базису sinc.
    """
    def set_config(self, config: _Config):
        """
        Метод замены конфигурационных данных.
 
        Параметры:
            config: конфигурация расчётной системы.
        """
        bndW = 0.5 / config.pixel_size
        eikz = (_torch.exp(1j * config.K * config.distance)**0.5)[:, None]
        sq2p = (2 / _np.pi)**0.5
        sqzk = ((2 * config.distance / config.K)**0.5)[:, None]
        xm  = (config.X[:, 0] - config.Y[:, 0])[None, ...]
        mu1 = -_np.pi * sqzk * bndW - xm / sqzk
        mu2 = _np.pi * sqzk * bndW - xm / sqzk
        S1, C1 = _fresnel(mu1 * sq2p)
        S2, C2 = _fresnel(mu2 * sq2p)
        operator = ((config.pixel_size / _np.pi) / sqzk * eikz
                  * _torch.exp(0.5j * xm**2 * config.K[:, None] / config.distance)
                  * (C2 - C1 - 1j * (S2 - S1)) / sq2p)
        self.__operator = _torch.tensor(_np.stack([_toeplitz(operator[i], operator[i]) for i in range(operator.size(0))]))[:, None, ...]
        
    def __call__(self,
                 field: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод распространения светового поля.
 
        Параметры:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после распространения.
        """
        return _fresnel_propagation(self.__operator[selection].to(config.device),
                                     field.cfloat().to(config.device))

class Propagator_fresnel_operator:
    """
    Класс оператора разложения светового поля по базису цилиндрических волн.
    """
    def set_config(self, config: _Config):
        """
        Метод замены конфигурационных данных.
 
        Параметры:
            config: конфигурация расчётной системы.
        """
        d = config.pixel_size / 2
        S1, C1 = _fresnel(((config.K / _np.pi / config.distance) ** 0.5)[:, None]
                          * (config.X[0] - (config.Y[0] - d))[None, ...])
        S2, C2 = _fresnel(((config.K / _np.pi / config.distance) ** 0.5)[:, None]
                          * (config.X[0] - (config.Y[0] + d))[None, ...])
        operator = (((C1 - C2) + 1j * (S1 - S2))
                    * ((_torch.exp(1j * config.K * config.distance) / 2j) ** 0.5)[:, None])
        self.__operator = _torch.tensor(_np.stack([_toeplitz(operator[i], operator[i]) for i in range(operator.size(0))]))[:, None, ...]
        
    def __call__(self,
                 field: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод распространения светового поля.
 
        Параметры:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после распространения.
        """
        return _fresnel_propagation(self.__operator[selection].to(config.device),
                                     field.cfloat().to(config.device))

def _fourier_propagation(operator: _torch.Tensor,
                          field: _torch.Tensor) -> _torch.Tensor:
    return _fft.ifft2(_fft.fft2(field) * operator)

def _fresnel_propagation(operator: _torch.Tensor,
                          field: _torch.Tensor) -> _torch.Tensor:
    return operator @ field @ operator
import torch as _torch
import numpy as _np
from abc import ABC as _ABC
from typing import Tuple as _Tuple
import collections as _collections
import pickle as _pickle

class ConfigOpticBase(_ABC):
    """
    Абстрактный класс базовой информации о работе опттической установки.

    Поля:
        wavelength: длина волны, используемая в оптической установке.
        K: волновое число.
        distance: дистанция распространения светового поля между плоскостями.
    """
    def __init__(self, wavelength: float, distance: float):
        """
        Конструктор класса базовой информации о работе опттической установки.

        Args:
            wavelength: длина волны, используемая в оптической установке.
            distance: дистанция распространения светового поля между плоскостями.
        """
        self._wavelength: float = wavelength
        self._K: float = 2 * _np.pi / wavelength
        self._distance = distance

    @property
    def wavelength(self) -> float:
        """
        Returns: длинна волны.
        """
        return self._wavelength

    @property
    def K(self) -> float:
        """
        Returns: волновое число.
        """
        return self._K
    
    @property
    def distance(self) -> float:
        """
        Returns: дистанция распространения.
        """
        return self._distance

class ConfigDesignPlane:
    """
    Класс данных о расчётной плоскости.
    """
    def __init__(self,
                 pixel_count: None | int | _Tuple[int, int] = None,
                 pixel_size: None | float | _Tuple[float, float] = None,
                 aperture: None | float | _Tuple[float, float] = None
                ):
        """
        Конструктор класса данных о расчётной плоскости.

        Args:
            pixel_count: данные о кол-ве пикселей в измерении.
            pixel_size: данные о размере пикселей в измерении.
            aperture: данные о апертуре измерения.

        Note:
            1. Достаточно указать два из трёх входных аргумента.
            2. Если было переданно одно число, будет считаться, что значение у каждого измерения совпадают,
            если было переданно два числа, то первое число будет ассоциированно с Y измерением, второе с X.
        """
        if aperture is None:
            aperture = ConfigDesignPlane.__get_excluded_third(pixel_count, pixel_size, False)
        elif pixel_count is None:
            pixel_count = ConfigDesignPlane.__get_excluded_third(aperture, pixel_size)
        elif pixel_size is None:
            pixel_size = ConfigDesignPlane.__get_excluded_third(aperture, pixel_count)            

        self._pixel_count: int | _Tuple[int, int] = pixel_count
        self._pixel_size: float | _Tuple[float, float] = pixel_size
        self._aperture: float | _Tuple[float, float] = aperture

    @staticmethod
    def __get_excluded_third(first_element, second_element, div: bool = True):
        ConfigDesignPlane.__check_for_none(first_element, second_element)
        first_element_y, first_element_x = ConfigDesignPlane.__return_tuple(first_element)
        second_element_y, second_element_x = ConfigDesignPlane.__return_tuple(second_element)
        if div:
            return first_element_y / second_element_y, first_element_x / second_element_x
        else:
            return first_element_y * second_element_y, first_element_x * second_element_x
    
    @staticmethod
    def __check_for_none(first_element, second_element):
        if (first_element is None) or (second_element is None):
            raise TypeError("One of the provided elements is None, it is not possible to obtain the full dimensions of the calculated plane.")

    @staticmethod
    def __return_element(element: int | _Tuple[int, int], dim: int = 0):
        if isinstance(element, _collections.abc.Sequence):
            return element[dim]
        else:
            return element

    @staticmethod
    def __return_tuple(element: int | _Tuple[int, int]):
        if isinstance(element, _collections.abc.Sequence):
            return element
        else:
            return element, element

    @staticmethod
    def __get_linspace_by_dim(aperture, pixel_count):
        linspace = _torch.linspace(-aperture / 2, aperture / 2, pixel_count + 1)[:pixel_count]
        linspace += aperture / (2 * pixel_count)
        return linspace

    @property
    def pixel_count_by_x(self) -> float:
        """
        Returns:
            Информация о кол-ве пикселей по оси X.
        """
        return ConfigDesignPlane.__return_element(self._pixel_count, 1)
    @property
    def pixel_count_by_y(self) -> float:
        """
        Returns:
            Информация о кол-ве пикселей по оси Y.
        """
        return ConfigDesignPlane.__return_element(self._pixel_count)
    @property
    def pixel_count(self) -> _Tuple[float, float]:
        """
        Returns:
            Информация о кол-ве пикселей по каждой оси [Y, x].
        """
        return ConfigDesignPlane.__return_tuple(self._pixel_count)
    @property
    def pixel_size_by_x(self) -> int:
        """
        Returns:
            Информация о размере пикселей по оси X.
        """
        return ConfigDesignPlane.__return_element(self._pixel_size, 1)
    @property
    def pixel_size_by_y(self) -> int:
        """
        Returns:
            Информация о размере пикселей по оси Y.
        """
        return ConfigDesignPlane.__return_element(self._pixel_size)
    @property
    def pixel_size(self) -> _Tuple[int, int]:
        """
        Returns:
            Информация о размере пикселей по каждой оси [Y, x].
        """
        return ConfigDesignPlane.__return_tuple(self._pixel_size)
    @property
    def aperture_width(self) -> float:
        """
        Returns:
            Информация о ширине расчётной плоскасти.
        """
        return ConfigDesignPlane.__return_element(self._aperture, 1)
    @property
    def aperture_height(self) -> float:
        """
        Returns:
            Информация о высоте расчётной плоскасти.
        """
        return ConfigDesignPlane.__return_element(self._aperture)
    @property
    def aperture(self) -> _Tuple[float, float]:
        """
        Returns:
            Информация о высоте и ширине расчётной плоскасти [H, W].
        """
        return ConfigDesignPlane.__return_tuple(self._aperture)
    @property
    def linspace_by_x(self) -> _torch.Tensor:
        """
        Returns:
            Расчётная сетка по оси X.
        """
        return ConfigDesignPlane.__get_linspace_by_dim(self.aperture_width, self.pixel_count_by_x)
    @property
    def linspace_by_y(self) -> _torch.Tensor:
        """
        Returns:
            Расчётная сетка по оси Y.
        """
        return ConfigDesignPlane.__get_linspace_by_dim(self.aperture_height, self.pixel_count_by_y)
    @property
    def meshgrid(self) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Returns:
            Расчётная сетка по осям [Y, X].
        """
        linspace_by_x = self.linspace_by_x
        linspace_by_y = self.linspace_by_y
        return _torch.meshgrid((linspace_by_y, linspace_by_x))

class ConfigModelBase(_ABC):
    """
    Абстрактный класс базовой информации об оптической установке.
    """
    def __init__(self,
                 input_vector_plane: ConfigDesignPlane,
                 first_lens_plane: ConfigDesignPlane,
                 matrix_plane: ConfigDesignPlane,
                 second_lens_plane: ConfigDesignPlane,
                 output_vector_plane: ConfigDesignPlane
                ):
        self._input_vector_plane: ConfigDesignPlane = input_vector_plane
        self._first_lens_plane: ConfigDesignPlane = first_lens_plane
        self._matrix_plane: ConfigDesignPlane = matrix_plane
        self._second_lens_plane: ConfigDesignPlane = second_lens_plane
        self._output_vector_plane: ConfigDesignPlane = output_vector_plane

    @property
    def input_vector_plane(self) -> ConfigDesignPlane:
        """
        Returns:
            Информация о расчётной плоскости входного вектора.
        """
        return self._input_vector_plane
    @property
    def first_lens_plane(self) -> ConfigDesignPlane:
        """
        Returns:
            Информация о расчётной плоскости первой скрещенной линзы.
        """
        return self._first_lens_plane
    @property
    def matrix_plane(self) -> ConfigDesignPlane:
        """
        Returns:
            Информация о расчётной плоскости элемента матрицы.
        """
        return self._matrix_plane
    @property
    def second_lens_plane(self) -> ConfigDesignPlane:
        """
        Returns:
            Информация о расчётной плоскости второй скрещенной линзы.
        """
        return self._second_lens_plane
    @property
    def output_vector_plane(self) -> ConfigDesignPlane:
        """
        Returns:
            Информация о расчётной плоскости выходного вектора оптической установки.
        """
        return self._output_vector_plane

class Config(ConfigOpticBase, ConfigModelBase):
    """
    Класс конфигурации, хранит полную информацию о расчётной системе.
    """
    def __init__(self,
                 right_matrix_count_columns: int,
                 right_matrix_count_rows: int,
                 right_matrix_width: float,
                 right_matrix_height: float,
                 min_height_gap: float,
                 right_matrix_split_x: int = 1,
                 right_matrix_split_y: int = 1,
                 left_matrix_split_x: int = 1,
                 left_matrix_split_y: int = 1,
                 result_matrix_split: int = 1,
                 camera_pixel_size: float = 3.6e-6,
                 wavelength: float = 532e-9,
                 distance: float = 0.03,
                 lens_pixel_size: float = 1.8e-6,
                 lens_size: int = 8192):
        """
        Конструктор класса.
 
        Args:
            right_matrix_count_columns: число столбцов в правой матрице, участвующей в операции матричного умножения.
            right_matrix_count_rows: число строк в правой матрице, участвующей в операции матричного умножения.
            right_matrix_width: ширина в метрах оптического элемента правой матрицы, участвующей в операции матричного умножения.
            right_matrix_height: высота в метрах оптического элемента правой матрицы, участвующей в операции матричного умножения.
            min_height_gap: минимально возможный зазор для отображения вектора левой матрицы, участвующей в операции матричного умножения.
            right_matrix_split_x: число дробления элементов правой матрицы по X (используется для более точного моделирования).
            right_matrix_split_y: число дробления элементов правой матрицы по Y (используется для более точного моделирования).
            left_matrix_split_x: число дробления элементов левой матрицы по X (используется для более точного моделирования).
            left_matrix_split_y: число дробления элементов левой матрицы по Y (используется для более точного моделирования).
            result_matrix_split: число дробления элементов результирующей матрицы (используется для более точного моделирования).
            camera_pixel_size: физический размер пикселя камеры, считывающей результирующее световое поле.
            wavelength: длины волн в метрах используемых в системе.
            distance: дистанция в метрах распространения светового поля между плоскостями.
            lens_pixel_size: размер пикселя в метрах скрещенных линз в оптической системе (нужен исключительно для моделирования).
            lens_size: размер скрещенных линз в метрах в оптической системе (нужен исключительно для моделирования).
        """
        ConfigOpticBase.__init__(self, wavelength, distance)

        config_plane_one = ConfigDesignPlane((left_matrix_split_y, left_matrix_split_x * right_matrix_count_rows),
                                             aperture=(min_height_gap, right_matrix_height)
                                            )
        config_plane_lens = ConfigDesignPlane(lens_size, lens_pixel_size)
        config_plane_three = ConfigDesignPlane((right_matrix_count_columns * left_matrix_split_x, right_matrix_count_rows * right_matrix_split_y),
                                             aperture=(right_matrix_width, right_matrix_height)
                                            )
        config_plane_five = ConfigDesignPlane((right_matrix_count_columns * result_matrix_split, 1),
                                              aperture=(right_matrix_width, camera_pixel_size)
                                             )
        ConfigModelBase.__init__(self,
                                 config_plane_one,
                                 config_plane_lens,
                                 config_plane_three,
                                 config_plane_lens,
                                 config_plane_five
                                )
        
        self._matrix_split_x: int = right_matrix_split_x
        self._matrix_split_y: int = right_matrix_split_y
        self._input_vector_split_x: int = left_matrix_split_x
        self._input_vector_split_y: int = left_matrix_split_y
        self._result_vector_split: int = result_matrix_split

    @property
    def matrix_split_x(self) -> int:
        """
        Returns:
            Информация о разбиении пикселей элементов матрицы по оси X.
        """
        return self._matrix_split_x
    @property
    def matrix_split_y(self) -> int:
        """
        Returns:
            Информация о разбиении пикселей элементов матрицы по оси Y.
        """
        return self._matrix_split_y
    @property
    def input_vector_split_x(self) -> int:
        """
        Returns:
            Информация о разбиении пикселей элементов входного вектора по оси X.
        """
        return self._input_vector_split_x
    @property
    def input_vector_split_y(self) -> int:
        """
        Returns:
            Информация о разбиении пикселей элементов входного вектора по оси Y.
        """
        return self._input_vector_split_y
    @property
    def result_vector_split(self) -> int:
        """
        Returns:
            Информация о разбиении пикселей элементов выходного вектора по оси Y.
        """
        return self._result_vector_split

    def save(self, filename: str = "config.pth"):
        """
        Метод сохранения параметров конфигурации в файл.
 
        Args:
            filename: название файла с параметрами конфигурации.
        """
        with open(filename, 'wb') as f:
            _pickle.dump(self, f, protocol=_pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str = "config.pth") -> 'Config':
        """
        Метод загрузки параметров конфигурации из файла.
 
        Args:
            filename: название файла с параметрами конфигурации.
        """
        with open(filename, 'rb') as f:
            return _pickle.load(f)

import torch as _torch
import numpy as _np

class Config:
    """
    Класс конфигурации, хранит информацию о расчётной системе.
 
    Поля:
        array_size: размер вычислительной области в пикселях.
        pixel_size: дискретизация вычислительной области.
        scale_by_x: коэффициент мастабирования пикселей по горизонтали.
        scale_by_y: коэффициент мастабирования пикселей по вертикали.
        wavelength: длины волн используемых в системе.
        K: волновые числа.
        aperture_size: апертура вычислительной области.
        vector_size: размер входного вектора в пикселях.
        distance: дистанция распространения светового поля между плоскостями.
        X: координаты абсциссы вычислительной области.
        Y: координаты ординаты вычислительной области.
        R: координаты радиуса вычислительной области.
    """
    def __init__(self, array_size: int = 4096,
                 pixel_size: float = 1.8e-6,
                 scale_by_x: int = 1,
                 scale_by_y: int = 1,
                 vector_size: int = 512,
                 wavelength: _torch.Tensor = _torch.Tensor([532e-9]),
                 distance: float = 0.03):
        """
        Конструктор класса.
 
        Args:
            array_size: размер вычислительной области в пикселях.
            pixel_size: дискретизация вычислительной области.
            scale_by_x: коэффициент мастабирования пикселей по горизонтали.
            scale_by_y: коэффициент мастабирования пикселей по вертикали.
            vector_size: размер входного вектора в пикселях.
            wavelength: длины волн используемых в системе.
            distance: дистанция распространения светового поля между плоскостями.
        """
        self.array_size: int = array_size
        self.pixel_size: float = pixel_size
        self.wavelength: _torch.Tensor = wavelength
        self.K = 2 * _np.pi / self.wavelength
        self.aperture_size: float = self.array_size * self.pixel_size
        self.scale_by_x: int = scale_by_x
        self.scale_by_y: int = scale_by_y
        self.vector_size: int = vector_size
        self.distance: float = distance
        x = _torch.linspace(-self.aperture_size / 2,
                            self.aperture_size / 2,
                            self.array_size + 1)[:self.array_size]
        x = x + self.pixel_size/2
        self.Y, self.X = _torch.meshgrid(x, x, indexing='ij')
        self.R = (self.X**2 + self.Y**2)**0.5

    def save(self, filename: str = "config.pth"):
        """
        Метод сохранения параметров конфигурации в файл.
 
        Args:
            filename: название файла с параметрами конфигурации.
        """
        _torch.save(self, filename)

    def load(self, filename: str = "config.pth"):
        """
        Метод загрузки параметров конфигурации из файла.
 
        Args:
            filename: название файла с параметрами конфигурации.
        """
        self = _torch.load(filename)
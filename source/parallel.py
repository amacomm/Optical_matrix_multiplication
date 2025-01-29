import torch as _torch
import torch.nn as _nn
from typing import List, Union, Any
from collections.abc import Iterator


class DataParallel(_nn.Module):
    """
    Класс параллельного вычисления модели на системе с множеством вычислителей.
 
    Поля:
        module: расчётная модель.
        devices: список задействующихся вычислителей.
        output_device: устройство, в которое будут записаны результаты вычислений.
    """
    def __init__(self, module: _nn.Module, devices: Union[None, List[Union[int, _torch.device]]] = None,
                 output_device: Union[int, _torch.device] = None) -> None:
        """
        Конструктор класса.
 
        Args:
            module: расчётная модель.
            devices: список задействующихся вычислителей.
            output_device: устройство, в которое будут записаны результаты вычислений.
        """
        super(DataParallel, self).__init__()

        if not _torch.cuda.is_available():
            raise EnvironmentError("cuda is not available.")
            return

        if not devices:
            devices = [_torch.device(x) for x in range(_torch.cuda.device_count())]

        if not output_device:
            output_device = devices[0]

        self.module = module
        self.devices = devices
        self.output_device = output_device

    def buffers(self, *inputs) -> Iterator[_torch.Tensor]:
        '''
        Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.
    
        Yields:
            torch.Tensor: module buffer
        '''
        return self.module.buffers(*inputs)

    def parameters(self, *inputs) -> Iterator[_nn.parameter.Parameter]:
        '''
        Return an iterator over module parameters.
    
        This is typically passed to an optimizer.
    
        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
    
        Yields:
            Parameter: module parameter
        '''
        return self.module.parameters(*inputs)

    def forward(self, input: _torch.Tensor, other: _torch.Tensor, **kwargs: Any) -> _torch.Tensor:
        '''
        Return an iterator over module parameters.
    
        This is typically passed to an optimizer.
    
        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
    
        Returns:
            Parameter: module parameter
        '''
        dim: int = 2
        if 'dim' in kwargs:
            dim = kwargs['dim']

        scattered_input = _nn.parallel.scatter(input, self.devices, dim)
        broadcasted_other = _nn.parallel.comm.broadcast(other, self.devices)
        replicas = [self.module.to(x) for x in self.devices]
        # replicas = _nn.parallel.replicate(self.module, self.devices)
        
        stacked_input = [(scattered_input[i],) + (broadcasted_other[i],) for i in range(len(replicas))]

        outputs = _nn.parallel.parallel_apply(replicas, stacked_input)

        return _nn.parallel.gather(outputs, self.output_device, dim)
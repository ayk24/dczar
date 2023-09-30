import operator
import warnings
from itertools import chain
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch.cuda._utils import _get_device_index
from torch.nn.modules.module import Module
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs


def _check_balance(device_ids: List[int]) -> None:
    """Check that the GPUs being used are reasonably balanced.
    This will emit a warning if the standard deviation of the memory or cores
    used across the selected GPUs is > 0.1. This is a simple heuristic to try
    and warn users about using badly unbalanced GPUs.
    Args:
        device_ids (list of int): device ids to check imbalance of
    """

    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class DataParallel(Module):
    r"""Implements data parallelism at the module level.
    This container parallelizes the application of the given :attr:`module` by
    splitting the input across the specified devices by chunking in the batch
    dimension (other objects will be copied once per device). In the forward
    pass, the module is replicated on each device, and each replica handles a
    portion of the input. During the backwards pass, gradients from each replica
    are summed into the original module.
    The batch size should be larger than the number of GPUs used.
    .. warning::
        It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
        instead of this class, to do multi-GPU training, even if there is only a single
        node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.
    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel but some types are specially handled. tensors will be
    **scattered** on dim specified (default 0). tuple, list and dict types will
    be shallow copied. The other types will be shared among different threads
    and can be corrupted if written to in the model's forward pass.
    The parallelized :attr:`module` must have its parameters and buffers on
    ``device_ids[0]`` before running this :class:`~torch.nn.DataParallel`
    module.
    .. warning::
        In each forward, :attr:`module` is **replicated** on each device, so any
        updates to the running module in ``forward`` will be lost. For example,
        if :attr:`module` has a counter attribute that is incremented in each
        ``forward``, it will always stay at the initial value because the update
        is done on the replicas which are destroyed after ``forward``. However,
        :class:`~torch.nn.DataParallel` guarantees that the replica on
        ``device[0]`` will have its parameters and buffers sharing storage with
        the base parallelized :attr:`module`. So **in-place** updates to the
        parameters or buffers on ``device[0]`` will be recorded. E.g.,
        :class:`~torch.nn.BatchNorm2d` and :func:`~torch.nn.utils.spectral_norm`
        rely on this behavior to update the buffers.
    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        will be invoked ``len(device_ids)`` times, each with inputs located on
        a particular device. Particularly, the hooks are only guaranteed to be
        executed in correct order with respect to operations on corresponding
        devices. For example, it is not guaranteed that hooks set via
        :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
        `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
        that each such hook be executed before the corresponding
        :meth:`~torch.nn.Module.forward` call of that device.
    .. warning::
        When :attr:`module` returns a scalar (i.e., 0-dimensional tensor) in
        :func:`forward`, this wrapper will return a vector of length equal to
        number of devices used in data parallelism, containing the result from
        each device.
    .. note::
        There is a subtlety in using the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
        details.
    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices (default: all devices)
        output_device (int or torch.device): device location of output (default: device_ids[0])
    Attributes:
        module (Module): the module to be parallelized
    Example::
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(
        self, module: Module, device_ids: Optional[List[int]] = None, output_device: Optional[int] = None, dim: int = 0
    ):
        """
        Initializes a DataParallel wrapper for a PyTorch module to enable parallel execution on multiple GPUs.

        Args:
            module (torch.nn.Module): The module to be parallelized.
            device_ids (List[int], optional): List of GPU device IDs to use for parallel execution. Defaults to None.
            output_device (int, optional): The GPU device ID where the final output will be gathered. Defaults to None.
            dim (int, optional): The dimension along which inputs will be split and distributed to devices. Defaults to 0.
        """
        super(DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """
        Executes the forward pass on the module on multiple GPUs.

        Args:
            *inputs (tuple): Variable-length input argument.
            **kwargs (dict): Keyword arguments.
        Returns:
            Any: The output of the forward pass.
        """
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj, t.device)
                )

        inputs = tuple(inputs[0])
        kwargs = tuple([{}] * len(inputs))
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module: Module, device_ids: List[int]) -> List[Module]:
        """
        Replicates the given module on specified GPUs.

        Args:
            module (torch.nn.Module): The module to be replicated.
            device_ids (List[int]): List of GPU device IDs.

        Returns:
            List[torch.nn.Module]: List of replicated modules.
        """
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs: Tuple, kwargs: dict, device_ids: List[int]) -> Tuple:
        """
        Splits inputs and keyword arguments and sends them to the specified GPUs.

        Args:
            inputs (Tuple): Input data to be scattered.
            kwargs (dict): Keyword arguments.
            device_ids (List[int]): List of GPU device IDs.

        Returns:
            Tuple: Scattered inputs and keyword arguments.
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(
        self,
        replicas: List[Module],
        inputs: Tuple,
        kwargs: Iterable[dict],
    ) -> List[Any]:
        """
        Applies the given inputs and keyword arguments to replicated modules in parallel.

        Args:
            replicas (List[torch.nn.Module]): List of replicated modules.
            inputs (Tuple): Input data to be applied.
            kwargs (Iterable[dict]): Iterable of keyword arguments.

        Returns:
            List[Any]: List of outputs from parallel application.
        """
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[: len(replicas)])

    def gather(self, outputs: List[Any], output_device: int) -> Any:
        """
        Gathers the output from multiple GPUs to the specified output device.

        Args:
            outputs (List[Any]): List of outputs from parallel processing.
            output_device (int): The GPU device ID where the final output will be gathered.

        Returns:
            Any: The gathered output.
        """
        return gather(outputs, output_device, dim=self.dim)

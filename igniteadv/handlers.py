from typing import Union, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import Engine, Events


class ModelGraphHandler(BaseHandler):
    """Helper handler to show model architecture on tensorboard.

    Args:
        model (torch.nn.Module): model to show
        data_loader (torch.utils.data.DataLoader): Data loader which supply training data to the model
        device (torch.device or str): Device name or torch device object
    """
    def __init__(self, model: nn.Module, data_loader: Optional[DataLoader], device: Optional[Union[str, torch.device]] = None) -> None:
        super(ModelGraphHandler, self).__init__()
        self._model = model
        self._device = device
        self._data_loader = data_loader

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'ModelGraphHandler' works only with TensorboardLogger")

        if event_name != Events.STARTED:
            raise RuntimeError("Handler 'ModelGraphHandler' only works at starting training")

        images, _ = next(iter(self._data_loader))
        if self._device is not None:
            images = images.to(self._device)
        logger.writer.add_graph(self._model, images)

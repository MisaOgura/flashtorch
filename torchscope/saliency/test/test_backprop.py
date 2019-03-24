import pdb
import pytest

import torchvision

from torchscope.saliency import Backprop


def test_set_model_to_eval_mode(mocker):
    mocked_models = mocker.spy(torchvision, 'models')
    model = mocked_models.vgg19()
    Backprop(model)

    model.eval.assert_called_once()

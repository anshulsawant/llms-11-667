import torch

from lm.generate import softmax_with_temperature
from pytest_utils.decorators import max_score


@max_score(5)
def test_softmax_with_temperature():
    logits = torch.FloatTensor([[-1.0, -2.0, -3.0], [3.0, 2.0, 1.0]])

    assert torch.allclose(
        softmax_with_temperature(logits, 0.0),
        torch.FloatTensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        rtol=1e-3,
    )

    assert torch.allclose(
        softmax_with_temperature(logits, 0.5),
        torch.FloatTensor([[0.86681, 0.11731, 0.01587], [0.86681, 0.11731, 0.01587]]),
        rtol=1e-3,
    )

    assert torch.allclose(
        softmax_with_temperature(logits, 1.0), logits.softmax(1), rtol=1e-3
    )

    assert torch.allclose(
        softmax_with_temperature(logits, 1e9),
        torch.FloatTensor(torch.ones((2, 3)) / 3),
        rtol=1e-3,
    )
from pytest_utils.decorators import max_score
import numpy as np
from bias.utils import read_data, eval_accuracy, calibrate, get_label_probs
from dataclasses import dataclass
from typing import Dict, List
import pytest
from unittest.mock import patch, MagicMock
import os


@dataclass
class Logprobs:
    text_offset: List[int]
    token_logprobs: List[float]
    tokens: List[str]
    top_logprobs: List[Dict[str, float]]

@dataclass
class CompletionChoice:
    finish_reason: str
    index: int
    logprobs: Logprobs
    text: str


# Test data
raw_responses_1 = [
    CompletionChoice(
        finish_reason='length',
        index=0,
        logprobs=Logprobs(
            text_offset=[270],
            token_logprobs=[-0.48199233],
            tokens=[' Positive'],
            top_logprobs=[{' Positive': -0.48199233}]
        ),
        text=' Positive'
    ),
    CompletionChoice(
        finish_reason='length',
        index=0,
        logprobs=Logprobs(
            text_offset=[244],
            token_logprobs=[-1.069571],
            tokens=[' Negative'],
            top_logprobs=[{' Positive': -0.72928005}]
        ),
        text=' Negative'
    )
]

raw_responses_2 = [
    CompletionChoice(
        finish_reason='length',
        index=0,
        logprobs=Logprobs(
            text_offset=[354],
            token_logprobs=[-0.18056089],
            tokens=[' Negative'],
            top_logprobs=[{' Negative': -0.18056089}]
        ),
        text=' Negative'
    ),
    CompletionChoice(
        finish_reason='length',
        index=0,
        logprobs=Logprobs(
            text_offset=[485],
            token_logprobs=[-0.7190584],
            tokens=[' Positive'],
            top_logprobs=[{' Positive': -0.7190584}]
        ),
        text=' Positive'
    )
]

few_shot_sentences_1 = ['they spend years trying to comprehend it ', 'half-assed ']
few_shot_labels_1 = [0, 0]
test_sentences_1 = ['it gets onto the screen just about as much of the novella as one could reasonably expect , and is engrossing and moving in its own right . ', 'my big fat greek wedding uses stereotypes in a delightful blend of sweet romance and lovingly dished out humor . ']
q_prefix_1 = "Review: "
a_prefix_1 = "Sentiment: "

few_shot_sentences_2 = [', it tends to speculation , conspiracy theories or , at best , circumstantial evidence ', "amusing enough while you watch it , offering fine acting moments and pungent insights into modern l.a. 's show-biz and media "]
few_shot_labels_2 = [0, 1]
test_sentences_2 = ['one of the more irritating cartoons you will see this , or any , year . ', "it 's a demented kitsch mess ( although the smeary digital video does match the muddled narrative ) , but it 's savvy about celebrity and has more guts and energy than much of what will open this year . "]
q_prefix_2 = "Comment: "
a_prefix_2 = "Label: "


class MockResponse:
    def __init__(self, choices):
        self.choices = choices

class MockOpenAI:
    def completions_create(self, **kwargs):
        model = kwargs.get('model')
        prompt = kwargs.get('prompt')
        echo = kwargs.get('echo', False)

        # Get the last sentiment (what we're actually testing)
        last_part = prompt.split('\n')[-1].strip()
        
        if 'N/A' in prompt:
            if "Negative" in last_part:
                return MockResponse([CompletionChoice(
                    finish_reason='length',
                    index=0,
                    logprobs=Logprobs(
                        text_offset=[0],
                        token_logprobs=[np.log(0.59237465)],
                        tokens=[' Negative'],
                        top_logprobs=[{' Negative': np.log(0.59237465)}]
                    ),
                    text=' Negative'
                )])
            elif 'Positive' in last_part:
                return MockResponse([CompletionChoice(
                    finish_reason='length',
                    index=0,
                    logprobs=Logprobs(
                        text_offset=[0],
                        token_logprobs=[np.log(0.40762535)],
                        tokens=[' Positive'],
                        top_logprobs=[{' Positive': np.log(0.40762535)}]
                    ),
                    text=' Positive'
                )])
        elif '[MASK]' in prompt:
            if "Negative" in last_part:
                return MockResponse([CompletionChoice(
                    finish_reason='length',
                    index=0,
                    logprobs=Logprobs(
                        text_offset=[0],
                        token_logprobs=[np.log(0.48789831)],
                        tokens=[' Negative'],
                        top_logprobs=[{' Negative': np.log(0.48789831)}]
                    ),
                    text=' Negative'
                )])
            elif 'Positive' in last_part:
                return MockResponse([CompletionChoice(
                    finish_reason='length',
                    index=0,
                    logprobs=Logprobs(
                        text_offset=[0],
                        token_logprobs=[np.log(0.51210169)],
                        tokens=[' Positive'],
                        top_logprobs=[{' Positive': np.log(0.51210169)}]
                    ),
                    text=' Positive'
                )])
        
        # Default response
        return MockResponse([CompletionChoice(
            finish_reason='length',
            index=0,
            logprobs=Logprobs(
                text_offset=[0],
                token_logprobs=[-1.0],
                tokens=[' Negative'],
                top_logprobs=[{' Negative': -1.0}]
            ),
            text=' Negative'
        )])

@pytest.fixture(autouse=True)
def mock_openai():
    with patch('openai.OpenAI') as mock:
        mock.return_value = MagicMock()
        mock.return_value.completions.create = MockOpenAI().completions_create
        yield mock


@max_score(1)
def test_read_data():
    train_s, train_l, test_s, test_l = read_data(42, 5, 2)
    assert train_s == ['klein , charming in comedies like american pie and dead-on in election , ', 'be fruitful ', 'soulful and ', 'the proud warrior that still lingers in the souls of these characters ', 'covered earlier and much better ']
    assert train_l == [1, 1, 1, 1, 0]
    assert test_s == ['it gets onto the screen just about as much of the novella as one could reasonably expect , and is engrossing and moving in its own right . ', 'my big fat greek wedding uses stereotypes in a delightful blend of sweet romance and lovingly dished out humor . ']
    assert test_l == [1, 1]


@max_score(4)
def test_eval_accuracy():
    # test case 1
    all_label_probs = np.array([
        [0.06393079, 0.77421383],
        [0.1667557, 0.70038244],
        [0.09027979, 0.72553299],
        [0.26096262, 0.57575094],
        [0.44449494, 0.29413527],
        [0.76972947, 0.11116619],
        [0.15828325, 0.69313992],
        [0.7516985, 0.12286153],
        [0.14077385, 0.5864067],
        [0.93894402, 0.00826436],
        [0.96010861, 0.00647536],
        [0.54544509, 0.28974226],
        [0.03732746, 0.86166848],
        [0.28822833, 0.57433105],
        [0.41491965, 0.41382738],
        [0.97500453, 0.00645249],
        [0.20891832, 0.62227116],
        [0.92885883, 0.03494045],
        [0.82968728, 0.07771125],
        [0.95564098, 0.00875388]
    ])
    test_labels = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]
    p_cf = np.array([0.56763354, 0.43236646])

    assert eval_accuracy(all_label_probs, test_labels) == 0.8
    assert eval_accuracy(all_label_probs, test_labels, p_cf) == 0.85

    # test case 2
    all_label_probs = np.array([
        [0.0765643, 0.6175518],
        [0.3431557, 0.48225607],
        [0.22774115, 0.58043486],
        [0.34067702, 0.49029753],
        [0.52253903, 0.23967175],
        [0.78737602, 0.08778082],
        [0.12911086, 0.62931019],
        [0.73348238, 0.11916208],
        [0.24077334, 0.41858033],
        [0.86582095, 0.01401659],
        [0.92564569, 0.0140888],
        [0.51724768, 0.27402682],
        [0.08901922, 0.70772813],
        [0.28172074, 0.53347882],
        [0.49782925, 0.27598689],
        [0.95588436, 0.0060631],
        [0.44738155, 0.35209812],
        [0.9329928, 0.02171223],
        [0.834215, 0.06879091],
        [0.90256673, 0.02255538]
    ])
    test_labels = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]
    p_cf = np.array([0.59237465, 0.40762535])

    assert eval_accuracy(all_label_probs, test_labels) == 0.75
    assert eval_accuracy(all_label_probs, test_labels, p_cf) == 0.8


@max_score(4)
def test_get_label_probs():
    assert np.allclose(
        get_label_probs(raw_responses_1, few_shot_sentences_1, few_shot_labels_1, test_sentences_1, q_prefix_1, a_prefix_1),
        np.array([[0.36787944, 0.6175518 ], [0.36787944, 0.48225607]]),
        rtol=1e-5
    )
    assert np.allclose(
        get_label_probs(raw_responses_2, few_shot_sentences_2, few_shot_labels_2, test_sentences_2, q_prefix_2, a_prefix_2),
        np.array([[0.83480185, 0.36787944], [0.36787944, 0.4872108]]),
        rtol=1e-5
    )

@max_score(4)
def test_calibrate():
    assert np.allclose(
        calibrate("N/A", few_shot_sentences_1, few_shot_labels_1, q_prefix_1, a_prefix_1),
        np.array([0.59237465, 0.40762535]),
        rtol=1e-5
    )
    assert np.allclose(
        calibrate("[MASK]", few_shot_sentences_2, few_shot_labels_2, q_prefix_2, a_prefix_2),
        np.array([0.48789831, 0.51210169]),
        rtol=1e-5
    )

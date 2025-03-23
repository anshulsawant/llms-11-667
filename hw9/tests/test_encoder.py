from retriever.modeling.encoder import EncoderModel
import torch
from pytest_utils.decorators import max_score


@max_score(1)
def test_pooling():
    last_hidden_state = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ]
    )

    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    output = EncoderModel.pooling(None, last_hidden_state, attention_mask)

    expected_output = torch.nn.functional.normalize(
        torch.tensor([[0.7, 0.8, 0.9], [0.6, 0.5, 0.4]]), p=2, dim=-1
    )

    assert torch.allclose(
        output, expected_output, atol=1e-5
    ), f"Expected {expected_output}, but got {output}"


@max_score(1)
def test_compute_similarity():
    q_reps = torch.tensor(
        [
            [1.0, 2.0, 1.0],
            [3.0, 1.0, 2.0],
        ]
    )

    p_reps = torch.tensor(
        [
            [5.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    )

    temperature = 0.01

    output = EncoderModel.compute_similarity(None, q_reps, p_reps, temperature)

    expected_output = torch.tensor([[1000.0, 400.0], [1900.0, 500.0]])

    assert torch.allclose(output, expected_output, atol=1e-5)


@max_score(1)
def test_compute_labels():
    n_queries = 3
    n_passages = 9

    output = EncoderModel.compute_labels(None, n_queries, n_passages)
    expected_output = torch.tensor([0, 3, 6], dtype=torch.long)

    assert torch.equal(output, expected_output)

    n_queries = 10
    n_passages = 100

    output = EncoderModel.compute_labels(None, n_queries, n_passages)
    expected_output = torch.tensor(
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=torch.long
    )

    assert torch.equal(output, expected_output)


@max_score(1)
def test_compute_loss():
    similarity_matrix = torch.tensor([[1000.0, 400.0], [1900.0, 500.0]])
    labels = torch.tensor([0, 1], dtype=torch.long)

    output = EncoderModel.compute_loss(None, similarity_matrix, labels)
    expected_output = torch.tensor(700.0)

    assert torch.allclose(output, expected_output, atol=1e-5)

    similarity_matrix = torch.tensor(
        [[0.81, 0.55, 0.74, 0.22], [0.55, 0.16, 0.91, 0.77]]
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    output = EncoderModel.compute_loss(None, similarity_matrix, labels)
    expected_output = torch.tensor(1.1464)

    assert torch.allclose(output, expected_output, atol=1e-4)

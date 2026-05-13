from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class ChoiceExample:
    candidate_features: list[list[float]]
    chosen_index: int
    weight: float = 1.0


@dataclass(frozen=True)
class RegressionExample:
    features: list[float]
    target: float
    weight: float = 1.0


class ScalarMLP:
    """A tiny dependency-free MLP that scores one action at a time."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        weights1: list[list[float]],
        biases1: list[float],
        weights2: list[float],
        bias2: float,
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if len(weights1) != hidden_dim:
            raise ValueError("weights1 must have one row per hidden unit")
        if len(biases1) != hidden_dim:
            raise ValueError("biases1 must have one entry per hidden unit")
        if len(weights2) != hidden_dim:
            raise ValueError("weights2 must have one entry per hidden unit")
        if any(len(row) != input_dim for row in weights1):
            raise ValueError("every weights1 row must match input_dim")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights1 = [list(row) for row in weights1]
        self.biases1 = list(biases1)
        self.weights2 = list(weights2)
        self.bias2 = float(bias2)

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        hidden_dim: int,
        seed: int,
    ) -> "ScalarMLP":
        rng = random.Random(seed)
        scale1 = math.sqrt(6.0 / (input_dim + hidden_dim))
        scale2 = math.sqrt(6.0 / (hidden_dim + 1))
        weights1 = [
            [rng.uniform(-scale1, scale1) for _ in range(input_dim)]
            for _ in range(hidden_dim)
        ]
        biases1 = [0.0 for _ in range(hidden_dim)]
        weights2 = [rng.uniform(-scale2, scale2) for _ in range(hidden_dim)]
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            weights1=weights1,
            biases1=biases1,
            weights2=weights2,
            bias2=0.0,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "ScalarMLP":
        return cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            weights1=payload["weights1"],
            biases1=payload["biases1"],
            weights2=payload["weights2"],
            bias2=float(payload["bias2"]),
        )

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "weights1": self.weights1,
            "biases1": self.biases1,
            "weights2": self.weights2,
            "bias2": self.bias2,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScalarMLP":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def _validate_features(self, features: Sequence[float]) -> None:
        if len(features) != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} features, got {len(features)}"
            )

    def _forward(self, features: Sequence[float]) -> tuple[list[float], float]:
        self._validate_features(features)
        hidden: list[float] = []
        for row, bias in zip(self.weights1, self.biases1):
            activation = bias + sum(weight * value for weight, value in zip(row, features))
            hidden.append(math.tanh(activation))
        score = self.bias2 + sum(
            weight * activation
            for weight, activation in zip(self.weights2, hidden)
        )
        return hidden, score

    def score(self, features: Sequence[float]) -> float:
        _, score = self._forward(features)
        return score

    def score_many(self, candidate_features: Iterable[Sequence[float]]) -> list[float]:
        return [self.score(features) for features in candidate_features]

    def predict_index(self, candidate_features: list[Sequence[float]]) -> int:
        if not candidate_features:
            raise ValueError("candidate_features must not be empty")
        scores = self.score_many(candidate_features)
        return max(range(len(scores)), key=lambda index: scores[index])

    def copy(self) -> "ScalarMLP":
        return ScalarMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            weights1=[list(row) for row in self.weights1],
            biases1=list(self.biases1),
            weights2=list(self.weights2),
            bias2=self.bias2,
        )

    def _assert_finite_parameters(self) -> None:
        values = [self.bias2, *self.biases1, *self.weights2]
        for row in self.weights1:
            values.extend(row)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("model parameters became non-finite during training")

    @staticmethod
    def _clip_scalar(value: float, clip_value: float | None) -> float:
        if clip_value is None or clip_value <= 0:
            return value
        return max(-clip_value, min(clip_value, value))

    def train_choice_examples(
        self,
        examples: list[ChoiceExample],
        *,
        epochs: int,
        learning_rate: float,
        l2: float = 0.0,
        seed: int = 0,
        gradient_clip: float | None = 5.0,
        progress_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if not examples:
            raise ValueError("at least one training example is required")

        rng = random.Random(seed)
        history: list[dict[str, float]] = []

        for epoch_index in range(epochs):
            shuffled_examples = list(examples)
            rng.shuffle(shuffled_examples)
            epoch_loss = 0.0
            epoch_correct = 0
            total_weight = 0.0

            for example in shuffled_examples:
                if not example.candidate_features:
                    raise ValueError("training examples must contain candidates")
                if not 0 <= example.chosen_index < len(example.candidate_features):
                    raise ValueError("chosen_index is out of range for example candidates")
                if example.weight <= 0:
                    raise ValueError("training example weight must be positive")

                hidden_by_candidate: list[list[float]] = []
                scores: list[float] = []
                for candidate_features in example.candidate_features:
                    hidden, score = self._forward(candidate_features)
                    hidden_by_candidate.append(hidden)
                    scores.append(score)

                predicted_index = max(range(len(scores)), key=lambda index: scores[index])
                if predicted_index == example.chosen_index:
                    epoch_correct += 1

                max_score = max(scores)
                exp_scores = [math.exp(score - max_score) for score in scores]
                denom = sum(exp_scores)
                probabilities = [value / denom for value in exp_scores]
                chosen_probability = max(probabilities[example.chosen_index], 1e-12)
                epoch_loss += example.weight * -math.log(chosen_probability)
                total_weight += example.weight

                grad_w1 = [
                    [0.0 for _ in range(self.input_dim)]
                    for _ in range(self.hidden_dim)
                ]
                grad_b1 = [0.0 for _ in range(self.hidden_dim)]
                grad_w2 = [0.0 for _ in range(self.hidden_dim)]
                grad_b2 = 0.0

                for candidate_index, candidate_features in enumerate(example.candidate_features):
                    grad_score = probabilities[candidate_index] * example.weight
                    if candidate_index == example.chosen_index:
                        grad_score -= example.weight

                    hidden = hidden_by_candidate[candidate_index]
                    for hidden_index in range(self.hidden_dim):
                        grad_w2[hidden_index] += grad_score * hidden[hidden_index]

                    grad_b2 += grad_score

                    for hidden_index in range(self.hidden_dim):
                        tanh_value = hidden[hidden_index]
                        grad_hidden = grad_score * self.weights2[hidden_index]
                        grad_pre_activation = grad_hidden * (1.0 - (tanh_value * tanh_value))
                        grad_b1[hidden_index] += grad_pre_activation

                        row_grad = grad_w1[hidden_index]
                        for input_index, feature_value in enumerate(candidate_features):
                            row_grad[input_index] += grad_pre_activation * feature_value

                for hidden_index in range(self.hidden_dim):
                    for input_index in range(self.input_dim):
                        if l2:
                            grad_w1[hidden_index][input_index] += (
                                l2 * self.weights1[hidden_index][input_index]
                            )
                        grad_w1[hidden_index][input_index] = self._clip_scalar(
                            grad_w1[hidden_index][input_index],
                            gradient_clip,
                        )
                        self.weights1[hidden_index][input_index] -= (
                            learning_rate * grad_w1[hidden_index][input_index]
                        )
                    if l2:
                        grad_w2[hidden_index] += l2 * self.weights2[hidden_index]
                    grad_b1[hidden_index] = self._clip_scalar(
                        grad_b1[hidden_index],
                        gradient_clip,
                    )
                    grad_w2[hidden_index] = self._clip_scalar(
                        grad_w2[hidden_index],
                        gradient_clip,
                    )
                    self.biases1[hidden_index] -= learning_rate * grad_b1[hidden_index]
                    self.weights2[hidden_index] -= learning_rate * grad_w2[hidden_index]

                grad_b2 = self._clip_scalar(grad_b2, gradient_clip)
                self.bias2 -= learning_rate * grad_b2
                self._assert_finite_parameters()

            epoch_metrics = {
                "epoch": float(epoch_index + 1),
                "loss": epoch_loss / max(total_weight, 1e-12),
                "accuracy": epoch_correct / len(shuffled_examples),
            }
            history.append(epoch_metrics)
            if progress_callback is not None:
                progress_callback(epoch_metrics)

        return history

    def train_regression_examples(
        self,
        examples: list[RegressionExample],
        *,
        epochs: int,
        learning_rate: float,
        l2: float = 0.0,
        seed: int = 0,
        gradient_clip: float | None = 5.0,
        progress_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if not examples:
            raise ValueError("at least one regression example is required")

        rng = random.Random(seed)
        history: list[dict[str, float]] = []

        for epoch_index in range(epochs):
            shuffled_examples = list(examples)
            rng.shuffle(shuffled_examples)
            epoch_loss = 0.0
            total_weight = 0.0

            for example in shuffled_examples:
                if example.weight <= 0:
                    raise ValueError("regression example weight must be positive")
                hidden, prediction = self._forward(example.features)
                error = prediction - example.target
                epoch_loss += example.weight * error * error
                total_weight += example.weight

                grad_w1 = [
                    [0.0 for _ in range(self.input_dim)]
                    for _ in range(self.hidden_dim)
                ]
                grad_b1 = [0.0 for _ in range(self.hidden_dim)]
                grad_w2 = [0.0 for _ in range(self.hidden_dim)]
                grad_b2 = 2.0 * error * example.weight

                for hidden_index in range(self.hidden_dim):
                    grad_w2[hidden_index] = grad_b2 * hidden[hidden_index]
                    grad_hidden = grad_b2 * self.weights2[hidden_index]
                    grad_pre_activation = grad_hidden * (
                        1.0 - (hidden[hidden_index] * hidden[hidden_index])
                    )
                    grad_b1[hidden_index] = grad_pre_activation
                    for input_index, feature_value in enumerate(example.features):
                        grad_w1[hidden_index][input_index] = (
                            grad_pre_activation * feature_value
                        )

                for hidden_index in range(self.hidden_dim):
                    for input_index in range(self.input_dim):
                        if l2:
                            grad_w1[hidden_index][input_index] += (
                                l2 * self.weights1[hidden_index][input_index]
                            )
                        grad_w1[hidden_index][input_index] = self._clip_scalar(
                            grad_w1[hidden_index][input_index],
                            gradient_clip,
                        )
                        self.weights1[hidden_index][input_index] -= (
                            learning_rate * grad_w1[hidden_index][input_index]
                        )
                    if l2:
                        grad_w2[hidden_index] += l2 * self.weights2[hidden_index]
                    grad_b1[hidden_index] = self._clip_scalar(
                        grad_b1[hidden_index],
                        gradient_clip,
                    )
                    grad_w2[hidden_index] = self._clip_scalar(
                        grad_w2[hidden_index],
                        gradient_clip,
                    )
                    self.biases1[hidden_index] -= learning_rate * grad_b1[hidden_index]
                    self.weights2[hidden_index] -= learning_rate * grad_w2[hidden_index]

                grad_b2 = self._clip_scalar(grad_b2, gradient_clip)
                self.bias2 -= learning_rate * grad_b2
                self._assert_finite_parameters()

            epoch_metrics = {
                "epoch": float(epoch_index + 1),
                "mse": epoch_loss / max(total_weight, 1e-12),
            }
            history.append(epoch_metrics)
            if progress_callback is not None:
                progress_callback(epoch_metrics)

        return history

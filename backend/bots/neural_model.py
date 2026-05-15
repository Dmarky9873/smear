from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEFAULT_TRAINING_BACKEND = "auto"
DEFAULT_TRAINER_BATCH_SIZE = 64
TRAINING_BACKEND_CHOICES = ("auto", "python", "torch")
_torch_module: Any | None = None
_torch_nn_module: Any | None = None
_torch_import_attempted = False


def _load_torch() -> tuple[Any | None, Any | None]:
    global _torch_import_attempted, _torch_module, _torch_nn_module
    if _torch_import_attempted:
        return _torch_module, _torch_nn_module

    _torch_import_attempted = True
    try:
        torch_module = importlib.import_module("torch")
    except ImportError:  # pragma: no cover - exercised when torch is unavailable
        _torch_module = None
        _torch_nn_module = None
        return None, None

    _torch_module = torch_module
    _torch_nn_module = torch_module.nn
    return _torch_module, _torch_nn_module


def _require_torch() -> tuple[Any, Any]:
    torch_module, nn_module = _load_torch()
    if torch_module is None or nn_module is None:
        raise ImportError("torch is required for the PyTorch trainer")
    return torch_module, nn_module


def is_torch_available() -> bool:
    torch_module, _ = _load_torch()
    return torch_module is not None


def resolve_training_backend(backend: str = DEFAULT_TRAINING_BACKEND) -> str:
    normalized_backend = backend.strip().lower()
    if normalized_backend not in TRAINING_BACKEND_CHOICES:
        raise ValueError(
            f"unsupported training backend {backend!r}; expected one of {TRAINING_BACKEND_CHOICES}"
        )
    if normalized_backend == "auto":
        return "torch" if is_torch_available() else "python"
    if normalized_backend == "torch" and not is_torch_available():
        raise ImportError(
            "PyTorch backend requested but torch is not installed. "
            "Install torch or use trainer backend 'python'."
        )
    return normalized_backend


def _configure_torch_runtime(*, torch_num_threads: int | None) -> None:
    torch_module, _ = _load_torch()
    if torch_module is None or torch_num_threads is None:
        return
    bounded_threads = max(1, int(torch_num_threads))
    try:
        torch_module.set_num_threads(bounded_threads)
    except RuntimeError:
        pass
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass


@dataclass(frozen=True)
class ChoiceExample:
    candidate_features: list[list[float]]
    chosen_index: int
    weight: float = 1.0
    target_distribution: list[float] | None = None


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

    def _overwrite_from(self, other: "ScalarMLP") -> None:
        self.weights1 = [list(row) for row in other.weights1]
        self.biases1 = list(other.biases1)
        self.weights2 = list(other.weights2)
        self.bias2 = other.bias2

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

    @staticmethod
    def _normalized_target_distribution(example: ChoiceExample) -> list[float]:
        candidate_count = len(example.candidate_features)
        if candidate_count <= 0:
            raise ValueError("training examples must contain candidates")

        if example.target_distribution is None:
            if not 0 <= example.chosen_index < candidate_count:
                raise ValueError("chosen_index is out of range for example candidates")
            distribution = [0.0 for _ in range(candidate_count)]
            distribution[example.chosen_index] = 1.0
            return distribution

        if len(example.target_distribution) != candidate_count:
            raise ValueError(
                "target_distribution must match the number of example candidates"
            )
        if any(value < 0.0 or not math.isfinite(value) for value in example.target_distribution):
            raise ValueError("target_distribution values must be finite and non-negative")

        total = sum(example.target_distribution)
        if total <= 0.0:
            raise ValueError("target_distribution must sum to a positive value")
        return [value / total for value in example.target_distribution]

    @staticmethod
    def _build_torch_module(model: "ScalarMLP"):
        torch, nn = _require_torch()

        class TorchScalarMLP(nn.Module):
            def __init__(self, *, input_dim: int, hidden_dim: int):
                super().__init__()
                self.hidden = nn.Linear(input_dim, hidden_dim)
                self.output = nn.Linear(hidden_dim, 1)

            def forward(self, features):
                hidden = torch.tanh(self.hidden(features))
                return self.output(hidden).squeeze(-1)

        module = TorchScalarMLP(
            input_dim=model.input_dim,
            hidden_dim=model.hidden_dim,
        )
        with torch.no_grad():
            module.hidden.weight.copy_(
                torch.tensor(model.weights1, dtype=torch.float32)
            )
            module.hidden.bias.copy_(
                torch.tensor(model.biases1, dtype=torch.float32)
            )
            module.output.weight.copy_(
                torch.tensor([model.weights2], dtype=torch.float32)
            )
            module.output.bias.copy_(
                torch.tensor([model.bias2], dtype=torch.float32)
            )
        return module

    @classmethod
    def _from_torch_module(cls, module) -> "ScalarMLP":
        torch, _ = _require_torch()
        with torch.no_grad():
            return cls(
                input_dim=module.hidden.in_features,
                hidden_dim=module.hidden.out_features,
                weights1=module.hidden.weight.detach().cpu().tolist(),
                biases1=module.hidden.bias.detach().cpu().tolist(),
                weights2=module.output.weight.detach().cpu().reshape(-1).tolist(),
                bias2=float(module.output.bias.detach().cpu().item()),
            )

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
        backend: str = DEFAULT_TRAINING_BACKEND,
        batch_size: int = DEFAULT_TRAINER_BATCH_SIZE,
        torch_num_threads: int | None = None,
    ) -> list[dict[str, float]]:
        resolved_backend = resolve_training_backend(backend)
        if resolved_backend == "torch":
            return self._train_choice_examples_torch(
                examples,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                seed=seed,
                gradient_clip=gradient_clip,
                progress_callback=progress_callback,
                batch_size=batch_size,
                torch_num_threads=torch_num_threads,
            )
        return self._train_choice_examples_python(
            examples,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
            gradient_clip=gradient_clip,
            progress_callback=progress_callback,
        )

    def _train_choice_examples_python(
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
                target_distribution = self._normalized_target_distribution(example)
                target_index = max(
                    range(len(target_distribution)),
                    key=lambda index: target_distribution[index],
                )

                hidden_by_candidate: list[list[float]] = []
                scores: list[float] = []
                for candidate_features in example.candidate_features:
                    hidden, score = self._forward(candidate_features)
                    hidden_by_candidate.append(hidden)
                    scores.append(score)

                predicted_index = max(range(len(scores)), key=lambda index: scores[index])
                if predicted_index == target_index:
                    epoch_correct += 1

                max_score = max(scores)
                exp_scores = [math.exp(score - max_score) for score in scores]
                denom = sum(exp_scores)
                probabilities = [value / denom for value in exp_scores]
                epoch_loss += example.weight * -sum(
                    target_probability
                    * math.log(max(probability, 1e-12))
                    for probability, target_probability in zip(
                        probabilities,
                        target_distribution,
                    )
                    if target_probability > 0.0
                )
                total_weight += example.weight

                grad_w1 = [
                    [0.0 for _ in range(self.input_dim)]
                    for _ in range(self.hidden_dim)
                ]
                grad_b1 = [0.0 for _ in range(self.hidden_dim)]
                grad_w2 = [0.0 for _ in range(self.hidden_dim)]
                grad_b2 = 0.0

                for candidate_index, candidate_features in enumerate(example.candidate_features):
                    grad_score = (
                        probabilities[candidate_index]
                        - target_distribution[candidate_index]
                    ) * example.weight

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

    def _train_choice_examples_torch(
        self,
        examples: list[ChoiceExample],
        *,
        epochs: int,
        learning_rate: float,
        l2: float = 0.0,
        seed: int = 0,
        gradient_clip: float | None = 5.0,
        progress_callback: Callable[[dict[str, float]], None] | None = None,
        batch_size: int = DEFAULT_TRAINER_BATCH_SIZE,
        torch_num_threads: int | None = None,
    ) -> list[dict[str, float]]:
        torch, _ = _require_torch()
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not examples:
            raise ValueError("at least one training example is required")

        _configure_torch_runtime(torch_num_threads=torch_num_threads)
        network = self._build_torch_module(self)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        example_count = len(examples)
        max_candidates = max(len(example.candidate_features) for example in examples)
        features = torch.zeros(
            (example_count, max_candidates, self.input_dim),
            dtype=torch.float32,
        )
        mask = torch.zeros((example_count, max_candidates), dtype=torch.bool)
        targets = torch.zeros((example_count, max_candidates), dtype=torch.float32)
        target_indices = torch.zeros(example_count, dtype=torch.long)
        example_weights = torch.zeros(example_count, dtype=torch.float32)

        for example_index, example in enumerate(examples):
            if not example.candidate_features:
                raise ValueError("training examples must contain candidates")
            if not 0 <= example.chosen_index < len(example.candidate_features):
                raise ValueError("chosen_index is out of range for example candidates")
            if example.weight <= 0:
                raise ValueError("training example weight must be positive")
            target_distribution = self._normalized_target_distribution(example)
            candidate_count = len(example.candidate_features)
            for candidate_index, candidate_features in enumerate(example.candidate_features):
                self._validate_features(candidate_features)
                features[example_index, candidate_index] = torch.tensor(
                    candidate_features,
                    dtype=torch.float32,
                )
            mask[example_index, :candidate_count] = True
            targets[example_index, :candidate_count] = torch.tensor(
                target_distribution,
                dtype=torch.float32,
            )
            target_indices[example_index] = int(
                max(
                    range(candidate_count),
                    key=lambda index: target_distribution[index],
                )
            )
            example_weights[example_index] = float(example.weight)

        rng = random.Random(seed)
        history: list[dict[str, float]] = []
        negative_inf = torch.finfo(torch.float32).min

        for epoch_index in range(epochs):
            shuffled_indices = list(range(example_count))
            rng.shuffle(shuffled_indices)
            epoch_loss = 0.0
            epoch_correct = 0
            total_weight = 0.0

            for batch_start in range(0, example_count, batch_size):
                batch_indices = shuffled_indices[batch_start : batch_start + batch_size]
                batch_features = features[batch_indices]
                batch_mask = mask[batch_indices]
                batch_targets = targets[batch_indices]
                batch_target_indices = target_indices[batch_indices]
                batch_weights = example_weights[batch_indices]

                optimizer.zero_grad()
                scores = network(batch_features.reshape(-1, self.input_dim)).reshape(
                    len(batch_indices),
                    max_candidates,
                )
                scores = scores.masked_fill(~batch_mask, negative_inf)
                log_probabilities = torch.log_softmax(scores, dim=1)
                losses = -(batch_targets * log_probabilities).sum(dim=1)
                weighted_loss = (losses * batch_weights).sum() / batch_weights.sum()
                if l2:
                    weighted_loss = weighted_loss + (0.5 * l2 * (
                        network.hidden.weight.square().sum()
                        + network.output.weight.square().sum()
                    ))
                weighted_loss.backward()
                if gradient_clip is not None and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
                optimizer.step()

                with torch.no_grad():
                    predictions = scores.argmax(dim=1)
                    epoch_correct += int((predictions == batch_target_indices).sum().item())
                    epoch_loss += float((losses * batch_weights).sum().item())
                    total_weight += float(batch_weights.sum().item())

            epoch_metrics = {
                "epoch": float(epoch_index + 1),
                "loss": epoch_loss / max(total_weight, 1e-12),
                "accuracy": epoch_correct / example_count,
            }
            history.append(epoch_metrics)
            if progress_callback is not None:
                progress_callback(epoch_metrics)

        self._overwrite_from(self._from_torch_module(network))
        self._assert_finite_parameters()
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
        backend: str = DEFAULT_TRAINING_BACKEND,
        batch_size: int = DEFAULT_TRAINER_BATCH_SIZE,
        torch_num_threads: int | None = None,
    ) -> list[dict[str, float]]:
        resolved_backend = resolve_training_backend(backend)
        if resolved_backend == "torch":
            return self._train_regression_examples_torch(
                examples,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                seed=seed,
                gradient_clip=gradient_clip,
                progress_callback=progress_callback,
                batch_size=batch_size,
                torch_num_threads=torch_num_threads,
            )
        return self._train_regression_examples_python(
            examples,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
            gradient_clip=gradient_clip,
            progress_callback=progress_callback,
        )

    def _train_regression_examples_python(
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

    def _train_regression_examples_torch(
        self,
        examples: list[RegressionExample],
        *,
        epochs: int,
        learning_rate: float,
        l2: float = 0.0,
        seed: int = 0,
        gradient_clip: float | None = 5.0,
        progress_callback: Callable[[dict[str, float]], None] | None = None,
        batch_size: int = DEFAULT_TRAINER_BATCH_SIZE,
        torch_num_threads: int | None = None,
    ) -> list[dict[str, float]]:
        torch, _ = _require_torch()
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not examples:
            raise ValueError("at least one regression example is required")

        _configure_torch_runtime(torch_num_threads=torch_num_threads)
        network = self._build_torch_module(self)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        example_count = len(examples)
        features = torch.zeros((example_count, self.input_dim), dtype=torch.float32)
        targets = torch.zeros(example_count, dtype=torch.float32)
        example_weights = torch.zeros(example_count, dtype=torch.float32)

        for example_index, example in enumerate(examples):
            if example.weight <= 0:
                raise ValueError("regression example weight must be positive")
            self._validate_features(example.features)
            features[example_index] = torch.tensor(example.features, dtype=torch.float32)
            targets[example_index] = float(example.target)
            example_weights[example_index] = float(example.weight)

        rng = random.Random(seed)
        history: list[dict[str, float]] = []

        for epoch_index in range(epochs):
            shuffled_indices = list(range(example_count))
            rng.shuffle(shuffled_indices)
            epoch_loss = 0.0
            total_weight = 0.0

            for batch_start in range(0, example_count, batch_size):
                batch_indices = shuffled_indices[batch_start : batch_start + batch_size]
                batch_features = features[batch_indices]
                batch_targets = targets[batch_indices]
                batch_weights = example_weights[batch_indices]

                optimizer.zero_grad()
                predictions = network(batch_features)
                losses = (predictions - batch_targets).square()
                weighted_loss = (losses * batch_weights).sum() / batch_weights.sum()
                if l2:
                    weighted_loss = weighted_loss + (0.5 * l2 * (
                        network.hidden.weight.square().sum()
                        + network.output.weight.square().sum()
                    ))
                weighted_loss.backward()
                if gradient_clip is not None and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
                optimizer.step()

                with torch.no_grad():
                    epoch_loss += float((losses * batch_weights).sum().item())
                    total_weight += float(batch_weights.sum().item())

            epoch_metrics = {
                "epoch": float(epoch_index + 1),
                "mse": epoch_loss / max(total_weight, 1e-12),
            }
            history.append(epoch_metrics)
            if progress_callback is not None:
                progress_callback(epoch_metrics)

        self._overwrite_from(self._from_torch_module(network))
        self._assert_finite_parameters()
        return history

"""End-to-end online CaMVo routing loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from camvo.aggregation import weighted_vote
from camvo.algorithm.calibration import BetaMixtureCalibrator, laplace_smooth
from camvo.algorithm.confidence import beta_cdf_confidence, exact_majority_confidence
from camvo.algorithm.linucb import LinUCBArm, LinUCBScore
from camvo.algorithm.oracle import ExhaustiveSubsetOracle, OracleCandidate, OracleSelection
from camvo.config import CaMVoConfig
from camvo.embeddings.base import EmbeddingProvider
from camvo.exceptions import CheckpointError, InsufficientResponsesError, ModelInvocationError
from camvo.llms.base import LLMClient
from camvo.types import AnnotationItem, ModelResponse, ModelScore, RoutingResult

_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class _ModelState:
    arm: LinUCBArm
    calibrator: BetaMixtureCalibrator
    observations: int = 0
    agreements: int = 0

    def historical_agreement(self, config: CaMVoConfig) -> float:
        numerator = self.agreements + config.agreement_prior_successes
        denominator = (
            self.observations
            + config.agreement_prior_successes
            + config.agreement_prior_failures
        )
        return numerator / denominator

    def state_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "agreements": self.agreements,
            "arm": self.arm.state_dict(),
            "calibrator": self.calibrator.state_dict(),
        }


class CaMVoRouter:
    """Stateful, online, auditable CaMVo classifier."""

    def __init__(
        self,
        models: list[LLMClient],
        embedder: EmbeddingProvider,
        config: CaMVoConfig,
    ) -> None:
        if not models:
            raise ValueError("at least one model is required")
        if len({model.model_id for model in models}) != len(models):
            raise ValueError("model ids must be unique")
        if config.min_models > len(models):
            raise ValueError("config.min_models exceeds the model pool size")
        if embedder.dimension != config.embedding_dim:
            raise ValueError("embedder dimension does not match config.embedding_dim")

        self.models = {model.model_id: model for model in models}
        self.embedder = embedder
        self.config = config
        self.round_index = 0
        self._states = {
            model_id: _ModelState(
                arm=LinUCBArm(
                    config.embedding_dim,
                    regularization=config.linucb_regularization,
                    exploration_alpha=config.exploration_alpha,
                    probability_epsilon=config.probability_epsilon,
                ),
                calibrator=BetaMixtureCalibrator(
                    epsilon=config.probability_epsilon,
                    min_samples_per_class=config.beta_min_samples_per_class,
                    max_concentration=config.beta_max_concentration,
                ),
            )
            for model_id in self.models
        }

        if config.confidence_method == "exact":
            confidence_function = exact_majority_confidence
        else:
            confidence_function = lambda lower, weights: beta_cdf_confidence(
                lower,
                weights,
                epsilon=config.probability_epsilon,
            )
        self._confidence_function = confidence_function
        self._oracle = ExhaustiveSubsetOracle(
            confidence_function,
            max_models=config.max_exhaustive_models,
        )

    def _context(self, item: AnnotationItem) -> np.ndarray:
        context = np.asarray(self.embedder.embed(item.text), dtype=np.float64)
        if context.shape != (self.config.embedding_dim,):
            raise ValueError("embedder returned an unexpected shape")
        if not np.all(np.isfinite(context)):
            raise ValueError("embedder returned non-finite values")
        return context

    def _score_models(
        self,
        item: AnnotationItem,
        context: np.ndarray,
        round_index: int,
    ) -> tuple[dict[str, ModelScore], dict[str, LinUCBScore]]:
        public_scores: dict[str, ModelScore] = {}
        bandit_scores: dict[str, LinUCBScore] = {}
        for model_id, model in self.models.items():
            state = self._states[model_id]
            bandit = state.arm.score(context)
            historical = state.historical_agreement(self.config)
            calibrated = state.calibrator.posterior(
                bandit.lower_confidence_score,
                historical,
            )
            lower_bound = laplace_smooth(
                calibrated,
                state.observations,
                round_index,
                self.config.laplace_regularization,
            )
            vote_weight = max(
                self.config.min_vote_weight,
                historical * bandit.prediction,
            )
            public_scores[model_id] = ModelScore(
                model_id=model_id,
                predicted_agreement=bandit.prediction,
                uncertainty=bandit.uncertainty,
                lcb_score=bandit.lower_confidence_score,
                calibrated_lower_bound=calibrated,
                smoothed_lower_bound=lower_bound,
                historical_agreement=historical,
                vote_weight=vote_weight,
                estimated_cost=model.estimate_cost(item),
            )
            bandit_scores[model_id] = bandit
        return public_scores, bandit_scores

    def _selection(
        self,
        scores: dict[str, ModelScore],
        warmup: bool,
    ) -> OracleSelection:
        candidates = [
            OracleCandidate(
                model_id=model_id,
                cost=score.estimated_cost,
                lower_bound=self._selection_lower_bound(score),
                vote_weight=score.vote_weight,
            )
            for model_id, score in scores.items()
        ]
        if not warmup:
            return self._oracle.select(
                candidates,
                threshold=self.config.confidence_threshold,
                min_models=self.config.min_models,
            )

        ordered = sorted(candidates, key=lambda candidate: candidate.model_id)
        confidence = self._confidence_function(
            [candidate.lower_bound for candidate in ordered],
            [candidate.vote_weight for candidate in ordered],
        )
        return OracleSelection(
            model_ids=tuple(candidate.model_id for candidate in ordered),
            confidence=confidence,
            cost=sum(candidate.cost for candidate in ordered),
            feasible=confidence >= self.config.confidence_threshold,
        )

    def _selection_lower_bound(self, score: ModelScore) -> float:
        """Confidence consumed by the Oracle; graph routers override this hook."""

        return score.smoothed_lower_bound

    @staticmethod
    def _validate_response(
        model_id: str,
        response: ModelResponse,
        item: AnnotationItem,
    ) -> None:
        if response.label not in item.labels:
            raise ValueError(
                f"model {model_id!r} returned {response.label!r}; expected one of {item.labels}"
            )
        if response.input_tokens < 0 or response.output_tokens < 0:
            raise ValueError("provider returned negative token usage")

    def _aggregate_responses(
        self,
        item: AnnotationItem,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        """Aggregate one pre-selected subset; graph routers may regularize this hook."""

        label, _ties = weighted_vote(
            responses,
            {model_id: scores[model_id].vote_weight for model_id in successful_ids},
            item.labels,
        )
        return label

    def _reward_reference_label(
        self,
        item: AnnotationItem,
        final_label: str,
        responses: dict[str, str],
        raw_responses: dict[str, ModelResponse],
        scores: dict[str, ModelScore],
        successful_ids: tuple[str, ...],
    ) -> str:
        """Consensus used to update model competence; extensions may separate heads."""

        return final_label

    def route(self, item: AnnotationItem) -> RoutingResult:
        """Route and label one item, then update online state."""

        next_round = self.round_index + 1
        context = self._context(item)
        scores, bandit_scores = self._score_models(item, context, next_round)
        warmup = next_round <= self.config.warmup_rounds
        selection = self._selection(scores, warmup)

        normalized_responses: dict[str, str] = {}
        raw_responses: dict[str, ModelResponse] = {}
        errors: dict[str, str] = {}
        for model_id in selection.model_ids:
            try:
                response = self.models[model_id].predict(item)
                self._validate_response(model_id, response, item)
                raw_responses[model_id] = response
                normalized_responses[model_id] = response.label
            except Exception as exc:  # provider boundaries must be isolated
                errors[model_id] = f"{type(exc).__name__}: {exc}"

        if errors and not self.config.allow_partial_responses:
            details = "; ".join(f"{key}: {value}" for key, value in errors.items())
            raise ModelInvocationError(f"one or more selected models failed: {details}")
        if len(normalized_responses) < self.config.min_models:
            raise InsufficientResponsesError(
                f"received {len(normalized_responses)} valid responses; "
                f"required at least {self.config.min_models}"
            )

        successful_ids = tuple(sorted(normalized_responses))
        if successful_ids != tuple(sorted(selection.model_ids)):
            confidence = self._confidence_function(
                [self._selection_lower_bound(scores[model_id]) for model_id in successful_ids],
                [scores[model_id].vote_weight for model_id in successful_ids],
            )
            estimated_cost = sum(scores[model_id].estimated_cost for model_id in successful_ids)
        else:
            confidence = selection.confidence
            estimated_cost = selection.cost

        label = self._aggregate_responses(
            item,
            normalized_responses,
            raw_responses,
            scores,
            successful_ids,
        )
        reward_reference_label = self._reward_reference_label(
            item,
            label,
            normalized_responses,
            raw_responses,
            scores,
            successful_ids,
        )

        should_update = len(successful_ids) > 1 or self.config.update_single_model_rounds
        if should_update:
            for model_id in successful_ids:
                matched = normalized_responses[model_id] == reward_reference_label
                state = self._states[model_id]
                state.arm.update(context, float(matched))
                state.calibrator.update(bandit_scores[model_id].prediction, matched)
                state.observations += 1
                state.agreements += int(matched)

        actual_cost = sum(
            self.models[model_id].pricing.cost(
                response.input_tokens,
                response.output_tokens,
            )
            for model_id, response in raw_responses.items()
        )
        self.round_index = next_round
        return RoutingResult(
            item_id=item.item_id,
            label=label,
            selected_models=successful_ids,
            subset_confidence=confidence,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            responses=normalized_responses,
            scores=scores,
            errors=errors,
            warmup=warmup,
        )

    def route_many(self, items: list[AnnotationItem]) -> list[RoutingResult]:
        return [self.route(item) for item in items]

    def observe_complete_feedback(
        self,
        item: AnnotationItem,
        responses: dict[str, ModelResponse],
        gold_label: str,
    ) -> None:
        """Warm-start the online learner from one fully audited response row.

        The method performs no provider calls.  It is intended for a formal
        calibration split whose complete response matrix has already been
        cached, and uses ground truth only before validation/test freezing.
        """

        if gold_label not in item.labels:
            raise ValueError("gold_label is outside the item's label space")
        if set(responses) != set(self.models):
            raise ValueError("complete feedback must contain the entire model pool")
        next_round = self.round_index + 1
        context = self._context(item)
        _scores, bandit_scores = self._score_models(item, context, next_round)
        for model_id in sorted(self.models):
            response = responses[model_id]
            self._validate_response(model_id, response, item)
            matched = response.label == gold_label
            state = self._states[model_id]
            state.arm.update(context, float(matched))
            state.calibrator.update(bandit_scores[model_id].prediction, matched)
            state.observations += 1
            state.agreements += int(matched)
        self.round_index = next_round

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "round_index": self.round_index,
            "config": self.config.to_dict(),
            "model_ids": sorted(self.models),
            "models": {
                model_id: self._states[model_id].state_dict() for model_id in sorted(self.models)
            },
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically save online state without serializing API credentials."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore state into an already configured router and provider pool."""

        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
            if int(state["schema_version"]) != _CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointError("unsupported checkpoint schema version")
            if state["config"] != self.config.to_dict():
                raise CheckpointError("checkpoint configuration does not match this router")
            if state["model_ids"] != sorted(self.models):
                raise CheckpointError("checkpoint model pool does not match this router")
            for model_id, model_state in state["models"].items():
                target = self._states[model_id]
                target.arm.load_state_dict(model_state["arm"])
                target.calibrator.load_state_dict(model_state["calibrator"])
                target.observations = int(model_state["observations"])
                target.agreements = int(model_state["agreements"])
            self.round_index = int(state["round_index"])
        except CheckpointError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid checkpoint: {exc}") from exc

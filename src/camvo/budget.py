"""Persistent, conservative experiment budget accounting."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from camvo.exceptions import BudgetExceededError, BudgetLedgerError

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    model_id: str
    item_id: str
    reserved_usd: float


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    hard_limit_usd: float
    max_provider_calls: int
    spent_usd: float
    reserved_usd: float
    remaining_usd: float
    provider_attempts: int
    completed_calls: int
    failed_calls: int
    cache_hits: int
    pending_reservations: int
    overrun_usd: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BudgetGuard:
    """Authorize provider calls and persist an auditable expense ledger.

    Reservations are written before network calls. If the process crashes,
    they remain pending and conservatively consume budget until explicitly
    reconciled. This prevents an automatic retry from silently double-spending.
    The guard is thread-safe within one process; do not share one ledger file
    between multiple processes.
    """

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        hard_limit_usd: float = 5.0,
        max_provider_calls: int = 500,
    ) -> None:
        if not math.isfinite(hard_limit_usd) or hard_limit_usd <= 0:
            raise ValueError("hard_limit_usd must be finite and positive")
        if max_provider_calls <= 0:
            raise ValueError("max_provider_calls must be positive")
        self.path = Path(ledger_path)
        self.hard_limit_usd = float(hard_limit_usd)
        self.max_provider_calls = int(max_provider_calls)
        self._lock = threading.RLock()
        self._state = self._load_or_initialize()

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "hard_limit_usd": self.hard_limit_usd,
            "max_provider_calls": self.max_provider_calls,
            "spent_usd": 0.0,
            "provider_attempts": 0,
            "completed_calls": 0,
            "failed_calls": 0,
            "cache_hits": 0,
            "overrun_usd": 0.0,
            "reservations": {},
            "calls": [],
        }

    def _load_or_initialize(self) -> dict[str, Any]:
        if not self.path.exists():
            state = self._new_state()
            self._persist(state)
            return state
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if int(state["schema_version"]) != _SCHEMA_VERSION:
                raise BudgetLedgerError("unsupported budget ledger schema")
            if not math.isclose(float(state["hard_limit_usd"]), self.hard_limit_usd):
                raise BudgetLedgerError("ledger hard limit does not match configuration")
            if int(state["max_provider_calls"]) != self.max_provider_calls:
                raise BudgetLedgerError("ledger call limit does not match configuration")
            if not isinstance(state["reservations"], dict) or not isinstance(
                state["calls"], list
            ):
                raise BudgetLedgerError("ledger collections are malformed")
            for key in (
                "spent_usd",
                "provider_attempts",
                "completed_calls",
                "failed_calls",
                "cache_hits",
                "overrun_usd",
            ):
                if key not in state:
                    raise BudgetLedgerError(f"ledger is missing {key}")
            for key in ("spent_usd", "overrun_usd"):
                value = float(state[key])
                if not math.isfinite(value) or value < 0:
                    raise BudgetLedgerError(f"ledger {key} must be finite and non-negative")
            for key in (
                "provider_attempts",
                "completed_calls",
                "failed_calls",
                "cache_hits",
            ):
                if int(state[key]) < 0:
                    raise BudgetLedgerError(f"ledger {key} must be non-negative")
            for reservation in state["reservations"].values():
                reserved = float(reservation["reserved_usd"])
                if not math.isfinite(reserved) or reserved < 0:
                    raise BudgetLedgerError("ledger reservation is invalid")
            return state
        except BudgetLedgerError:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise BudgetLedgerError(f"invalid budget ledger: {exc}") from exc

    def _persist(self, state: dict[str, Any] | None = None) -> None:
        payload = self._state if state is None else state
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _reserved_total(self) -> float:
        return sum(
            float(reservation["reserved_usd"])
            for reservation in self._state["reservations"].values()
        )

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            spent = float(self._state["spent_usd"])
            reserved = self._reserved_total()
            return BudgetSnapshot(
                hard_limit_usd=self.hard_limit_usd,
                max_provider_calls=self.max_provider_calls,
                spent_usd=spent,
                reserved_usd=reserved,
                remaining_usd=max(0.0, self.hard_limit_usd - spent - reserved),
                provider_attempts=int(self._state["provider_attempts"]),
                completed_calls=int(self._state["completed_calls"]),
                failed_calls=int(self._state["failed_calls"]),
                cache_hits=int(self._state["cache_hits"]),
                pending_reservations=len(self._state["reservations"]),
                overrun_usd=float(self._state["overrun_usd"]),
            )

    def reserve(
        self,
        *,
        model_id: str,
        item_id: str,
        estimated_max_cost_usd: float,
    ) -> BudgetReservation:
        if not math.isfinite(estimated_max_cost_usd) or estimated_max_cost_usd < 0:
            raise ValueError("estimated_max_cost_usd must be finite and non-negative")
        with self._lock:
            snapshot = self.snapshot()
            if snapshot.provider_attempts >= self.max_provider_calls:
                raise BudgetExceededError(
                    f"provider-call limit reached ({self.max_provider_calls})"
                )
            projected = snapshot.spent_usd + snapshot.reserved_usd + estimated_max_cost_usd
            if projected > self.hard_limit_usd + 1e-12:
                raise BudgetExceededError(
                    "provider call blocked before execution: "
                    f"projected ${projected:.6f} exceeds hard limit "
                    f"${self.hard_limit_usd:.6f}"
                )
            reservation = BudgetReservation(
                reservation_id=uuid.uuid4().hex,
                model_id=model_id,
                item_id=item_id,
                reserved_usd=float(estimated_max_cost_usd),
            )
            self._state["provider_attempts"] += 1
            self._state["reservations"][reservation.reservation_id] = {
                **asdict(reservation),
                "created_at_unix": time.time(),
            }
            self._persist()
            return reservation

    def commit(
        self,
        reservation: BudgetReservation,
        *,
        actual_cost_usd: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if not math.isfinite(actual_cost_usd) or actual_cost_usd < 0:
            raise ValueError("actual_cost_usd must be finite and non-negative")
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token usage must be non-negative")
        with self._lock:
            persisted = self._state["reservations"].pop(reservation.reservation_id, None)
            if persisted is None:
                raise BudgetLedgerError("reservation is unknown or already finalized")
            self._state["spent_usd"] += float(actual_cost_usd)
            self._state["completed_calls"] += 1
            self._state["overrun_usd"] = max(
                0.0,
                float(self._state["spent_usd"]) + self._reserved_total() - self.hard_limit_usd,
            )
            self._state["calls"].append(
                {
                    "status": "completed",
                    "reservation_id": reservation.reservation_id,
                    "model_id": reservation.model_id,
                    "item_id": reservation.item_id,
                    "reserved_usd": reservation.reserved_usd,
                    "actual_cost_usd": float(actual_cost_usd),
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "completed_at_unix": time.time(),
                }
            )
            self._persist()

    def fail(self, reservation: BudgetReservation, error: BaseException) -> None:
        with self._lock:
            persisted = self._state["reservations"].pop(reservation.reservation_id, None)
            if persisted is None:
                raise BudgetLedgerError("reservation is unknown or already finalized")
            self._state["failed_calls"] += 1
            self._state["calls"].append(
                {
                    "status": "failed",
                    "reservation_id": reservation.reservation_id,
                    "model_id": reservation.model_id,
                    "item_id": reservation.item_id,
                    "reserved_usd": reservation.reserved_usd,
                    "actual_cost_usd": 0.0,
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:500],
                    "failed_at_unix": time.time(),
                }
            )
            self._persist()

    def record_cache_hit(self) -> None:
        with self._lock:
            self._state["cache_hits"] += 1
            self._persist()

    def release_stale_reservation(self, reservation_id: str) -> None:
        """Manually release a known no-charge reservation after inspection."""

        with self._lock:
            persisted = self._state["reservations"].pop(reservation_id, None)
            if persisted is None:
                raise BudgetLedgerError("reservation does not exist")
            self._state["calls"].append(
                {
                    "status": "released",
                    "reservation_id": reservation_id,
                    "model_id": persisted["model_id"],
                    "item_id": persisted["item_id"],
                    "reserved_usd": persisted["reserved_usd"],
                    "actual_cost_usd": 0.0,
                    "released_at_unix": time.time(),
                }
            )
            self._persist()

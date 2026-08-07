"""Request-chain iteration budgets across gateway continuation turns."""

from agent.iteration_budget import (
    IterationBudget,
    REQUEST_CHAIN_BUDGET_EVENT_KEY,
)
from gateway.platforms.base import MessageEvent
from gateway.run import (
    _request_chain_budget_for_followup,
    _request_chain_budget_from_event,
)


def _event(*, internal: bool, budget: IterationBudget | None = None) -> MessageEvent:
    metadata = {}
    if budget is not None:
        metadata[REQUEST_CHAIN_BUDGET_EVENT_KEY] = budget
    return MessageEvent(text="continuation", internal=internal, metadata=metadata)


def test_internal_event_uses_its_originating_request_budget():
    current = IterationBudget(90)
    originating = IterationBudget(90)
    event = _event(internal=True, budget=originating)

    assert _request_chain_budget_from_event(event) is originating
    assert _request_chain_budget_for_followup(current, event) is originating


def test_legacy_internal_event_reuses_current_request_budget():
    current = IterationBudget(90)

    assert (
        _request_chain_budget_for_followup(current, _event(internal=True))
        is current
    )


def test_real_queued_user_message_starts_a_fresh_request_budget():
    current = IterationBudget(90)

    assert _request_chain_budget_from_event(_event(internal=False)) is None
    assert (
        _request_chain_budget_for_followup(current, _event(internal=False))
        is None
    )


def test_late_steer_without_an_event_remains_in_current_request():
    current = IterationBudget(90)

    assert (
        _request_chain_budget_for_followup(
            current,
            None,
            leftover_steer=True,
        )
        is current
    )

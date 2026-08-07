from agent import response_hooks


def test_max_pre_response_nudges_defaults():
    assert response_hooks.max_pre_response_nudges({}) == 2
    assert response_hooks.max_pre_response_nudges({"agent": {}}) == 2


def test_max_pre_response_nudges_accepts_string_and_clamps_negative():
    assert (
        response_hooks.max_pre_response_nudges(
            {"agent": {"max_pre_response_nudges": "4"}}
        )
        == 4
    )
    assert (
        response_hooks.max_pre_response_nudges(
            {"agent": {"max_pre_response_nudges": -1}}
        )
        == 0
    )

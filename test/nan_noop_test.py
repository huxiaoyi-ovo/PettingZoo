from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils.wrappers import NanNoopParallelV1, NanNoopV1

from .scale_action_test import DummyAEC, DummyParallel


NO_OP = np.array([0.25, 0.5, 0.75], dtype=np.float32)


def test_aec_replaces_array_nan_with_noop():
    inner = DummyAEC()
    env = NanNoopV1(inner, NO_OP)
    env.reset()
    agent = env.agent_selection

    with pytest.warns(UserWarning, match="NaN"):
        env.step(np.array([np.nan, 0.0, 0.0], dtype=np.float32))

    np.testing.assert_array_equal(inner.received[agent], NO_OP)


def test_aec_replaces_scalar_nan_with_noop():
    inner = DummyAEC(lambda agent: Discrete(5))
    env = NanNoopV1(inner, 0)
    env.reset()
    agent = env.agent_selection

    with pytest.warns(UserWarning, match="NaN"):
        env.step(float("nan"))

    assert inner.received[agent] == 0


def test_aec_passes_clean_action_through_unchanged():
    inner = DummyAEC()
    env = NanNoopV1(inner, NO_OP)
    env.reset()
    agent = env.agent_selection
    action = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    env.step(action)

    assert inner.received[agent] is action


def test_parallel_replaces_each_nan_agent_independently():
    inner = DummyParallel()
    env = NanNoopParallelV1(inner, NO_OP)
    env.reset()
    clean_0 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    clean_1 = np.array([0.4, 0.5, 0.6], dtype=np.float32)

    with pytest.warns(UserWarning, match="NaN"):
        env.step(
            {
                "agent_0": np.array([np.nan, 0.0, 0.0], dtype=np.float32),
                "agent_1": clean_1,
            }
        )
    np.testing.assert_array_equal(inner.received["agent_0"], NO_OP)
    assert inner.received["agent_1"] is clean_1

    with pytest.warns(UserWarning, match="NaN"):
        env.step(
            {
                "agent_0": clean_0,
                "agent_1": np.array([0.0, np.nan, 0.0], dtype=np.float32),
            }
        )
    assert inner.received["agent_0"] is clean_0
    np.testing.assert_array_equal(inner.received["agent_1"], NO_OP)


@pytest.mark.parametrize(
    ("wrapper_cls", "env_cls"),
    [(NanNoopV1, DummyAEC), (NanNoopParallelV1, DummyParallel)],
)
def test_incompatible_noop_is_rejected(wrapper_cls, env_cls):
    invalid_for_agent_1 = np.full(3, -0.5, dtype=np.float32)

    with pytest.raises(ValueError, match="agent_1"):
        wrapper_cls(env_cls(), invalid_for_agent_1)


def test_aec_dead_agent_none_action_passes_through():
    inner = DummyAEC()
    env = NanNoopV1(inner, NO_OP)
    env.reset()
    agent = env.agent_selection
    inner.terminations[agent] = True

    env.step(None)

    assert agent not in inner.received


def test_aec_api():
    api_test(NanNoopV1(DummyAEC(lambda agent: Discrete(5)), 0), num_cycles=5)


def test_parallel_api():
    parallel_api_test(
        NanNoopParallelV1(DummyParallel(lambda agent: Discrete(5)), 0),
        num_cycles=5,
    )

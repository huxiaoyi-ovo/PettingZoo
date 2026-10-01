"""Wrappers that replace NaN actions with a caller-supplied no-op action."""

from __future__ import annotations

import warnings
from typing import Any

import gymnasium.spaces
import numpy as np
from typing_extensions import override

from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType, ParallelEnv
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


def _validate_noop(
    space: gymnasium.spaces.Space[Any],
    no_op_action: Any,
    wrapper_name: str,
    agent: AgentID,
) -> None:
    """Raise if the no-op action is not valid for the agent's action space."""
    if not space.contains(no_op_action):
        raise ValueError(
            f"{wrapper_name} received no_op_action {no_op_action!r}, which is not "
            f"contained in the action space for agent {agent!r}: {space}."
        )


def _contains_nan(action: Any) -> bool:
    """Return whether a numeric scalar or array-like action contains a NaN."""
    if action is None:
        return False
    try:
        return bool(np.isnan(action).any())
    except TypeError:
        return False


def _replace_nan(action: Any, no_op_action: Any, env: Any) -> Any:
    """Replace an action containing NaN with the caller-supplied no-op."""
    if not _contains_nan(action):
        return action
    warnings.warn(
        f"Step received a NaN action {action}. Environment is {env}. "
        f"Taking the no-op action {no_op_action}.",
        stacklevel=3,
    )
    return no_op_action


class NanNoopV1(BaseWrapper[AgentID, ObsType, Any]):
    """Replace a numeric action containing a NaN with a caller-supplied no-op action.

    The same no_op_action is used for every agent wrapped by this instance.
    It must be contained in each affected agent's action space. Actions without
    a NaN and the None action of a dead AEC agent are passed through unchanged.
    A warning is emitted whenever a replacement occurs.

    :param env: The AEC environment to wrap.
    :param no_op_action: The action to send when an input action contains NaN.
    """

    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
        no_op_action: Any,
    ):
        assert isinstance(env, AECEnv), (
            "NanNoopV1 is only compatible with AEC environments, "
            "use NanNoopParallelV1 instead."
        )
        super().__init__(env)
        self.no_op_action = no_op_action
        for agent in getattr(env, "possible_agents", []):
            _validate_noop(
                self.env.action_space(agent), no_op_action, "NanNoopV1", agent
            )

    @override
    def step(self, action: Any) -> None:
        if action is None:
            self.env.step(None)
            return
        agent = self.agent_selection
        _validate_noop(
            self.env.action_space(agent), self.no_op_action, "NanNoopV1", agent
        )
        self.env.step(_replace_nan(action, self.no_op_action, self))

    @override
    def __str__(self) -> str:
        return f"NanNoopV1<{self.env!s}>"


class NanNoopParallelV1(BaseParallelWrapper[AgentID, ObsType, Any]):
    """Replace numeric actions containing a NaN with a caller-supplied no-op action.

    Each agent's action is checked independently, so only agents that send a NaN
    receive the replacement. The same no_op_action is used for every agent
    wrapped by this instance and must be contained in each affected agent's
    action space. Actions without a NaN are passed through unchanged. A warning
    is emitted whenever a replacement occurs.

    :param env: The parallel environment to wrap.
    :param no_op_action: The action to send when an input action contains NaN.
    """

    def __init__(
        self,
        env: ParallelEnv[AgentID, ObsType, ActionType],
        no_op_action: Any,
    ):
        super().__init__(env)
        self.no_op_action = no_op_action
        for agent in getattr(env, "possible_agents", []):
            _validate_noop(
                self.env.action_space(agent),
                no_op_action,
                "NanNoopParallelV1",
                agent,
            )

    @override
    def step(
        self, actions: dict[AgentID, Any]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        replaced: dict[AgentID, Any] = {}
        for agent, action in actions.items():
            _validate_noop(
                self.env.action_space(agent),
                self.no_op_action,
                "NanNoopParallelV1",
                agent,
            )
            replaced[agent] = _replace_nan(action, self.no_op_action, self)
        return self.env.step(replaced)

    @override
    def __str__(self) -> str:
        return f"NanNoopParallelV1<{self.env!s}>"

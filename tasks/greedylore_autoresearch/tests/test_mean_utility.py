"""Both language tasks contribute signed changes; vision stays a separate gate."""
# ruff: noqa: S101
import math

import pytest
from greedylore_autoresearch_grader.utility import advisory_utility
from test_utility import reference


@pytest.mark.parametrize('losses,delta', [((2.9, 4.05), -.025), ((3.1, 4.1), .1), ((2.9, 3.9), -.1)])
def test_mean_utility_rewards_improvements_and_averages_both_tasks(losses, delta):
    result = advisory_utility(dict(zip(('llama60m_adamw', 'llama130m_muon'), losses, strict=True)),
        {'130m_adamw_bf16': .1, '130m_adamw_fp32': .2}, reference())
    assert result['version'] == 'mean-log-ppl-ratio-v2'
    assert result['mean_loss_delta'] == pytest.approx(delta)
    assert result['value'] == pytest.approx(1.1 * math.exp(-delta))
    assert result['advisory_only']


def test_legacy_utility_remains_explicitly_available():
    result = advisory_utility({'llama60m_adamw': 2.9, 'llama130m_muon': 4.05},
        {'130m_adamw_bf16': .1, '130m_adamw_fp32': .2}, reference(), version='worst-ppl-discount-v1')
    assert result['value'] == pytest.approx(1.1 * math.exp(-.05))
    assert result['loss_penalty'] == pytest.approx(.05)

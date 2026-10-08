# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Refined-recompute probe: numerics unchanged, memory matches the frame."""

import tempfile
import types
import unittest

import paddle
from paddle.distributed.fleet.utils import recompute

from paddlefleet.recompute_utils import make_refined_recompute
from paddlefleet.refined_recompute import AutoRefinedRecompute
from paddlefleet.refined_recompute.probe import PROBE, RRProbe

_SEED = 42
_BATCH, _SEQ, _WIDE = 2, 8, 64
_POINT = "mla_o_proj"


def _config(**overrides):
    defaults = {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "recompute_modules": [],
        "virtual_pipeline_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class _Block(paddle.nn.Layer):
    """One RR point wired like a real call site."""

    def __init__(self, config, multi_op=False, layer_number=1):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.is_mtp_layer = False
        self.multi_op = multi_op
        self.up = paddle.nn.Linear(_WIDE, 4 * _WIDE, bias_attr=False)
        self.down = paddle.nn.Linear(4 * _WIDE, _WIDE, bias_attr=False)
        self.calls = 0
        self.use_rr, self._rr = make_refined_recompute(self, _POINT)

    def _boundary(self, hidden):
        self.calls += 1
        hidden = self.up(hidden)
        if self.multi_op:
            hidden = paddle.nn.functional.silu(hidden)
        return self.down(hidden)

    def forward(self, hidden):
        if self.use_rr and self.training:
            hidden = self._rr(self._boundary, hidden)
        else:
            hidden = self._boundary(hidden)
        return paddle.tanh(hidden)


def _build(multi_op=False, **overrides):
    paddle.seed(_SEED)
    block = _Block(_config(**overrides), multi_op=multi_op)
    block.train()
    return block


def _run(block):
    paddle.seed(_SEED)
    hidden = paddle.randn([_BATCH, _SEQ, _WIDE])
    hidden.stop_gradient = False
    output = recompute(block, hidden)
    weights = paddle.arange(1, output.shape[-1] + 1, dtype=output.dtype)
    (output * weights).sum().backward()
    grads = {
        name: param.grad.detach()
        for name, param in block.named_parameters()
        if param.grad is not None
    }
    return output.detach(), hidden.grad.detach(), grads


def _retained_bytes(block, hidden):
    paddle.device.synchronize()
    before = paddle.device.cuda.memory_allocated()
    with paddle.no_grad():
        output = block(hidden)
    del output
    paddle.device.synchronize()
    return paddle.device.cuda.memory_allocated() - before


class _ProbeCase(unittest.TestCase):
    def setUp(self):
        PROBE.deactivate()
        self.addCleanup(PROBE.deactivate)

    def activate(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        PROBE.activate(directory.name)


class TestProbeIsTransparent(_ProbeCase):
    def test_matches_plain_recompute_bitwise(self):
        baseline = _build()
        self.activate()
        probed = _build()
        probed.set_state_dict(baseline.state_dict())
        self.assertIsInstance(probed._rr, RRProbe)

        want = _run(baseline)
        got = _run(probed)

        self.assertEqual(baseline.calls, 2)
        self.assertEqual(probed.calls, 2)
        for name, expected, actual in zip(
            ("output", "input grad"), want[:2], got[:2], strict=True
        ):
            self.assertTrue(
                paddle.equal_all(expected, actual).item(),
                f"{name} mismatch: max diff "
                f"{(expected - actual).abs().max().item()}",
            )
        self.assertEqual(sorted(want[2]), sorted(got[2]))
        for key in want[2]:
            self.assertTrue(
                paddle.equal_all(want[2][key], got[2][key]).item(),
                f"grad of {key} mismatch",
            )


class TestOnlyRecomputePassIsPriced(_ProbeCase):
    def test_one_record_per_step(self):
        self.activate()
        block = _build()
        for step in range(1, 4):
            _run(block)
            self.assertEqual(len(PROBE._records), step)
            self.assertEqual(PROBE._records[-1][1], step - 1)
        keys = {record[0] for record in PROBE._records}
        self.assertEqual(keys, {(_POINT, 1, False)})

    def test_first_forward_alone_records_nothing(self):
        self.activate()
        block = _build()
        hidden = paddle.randn([_BATCH, _SEQ, _WIDE])
        with paddle.no_grad():
            block(hidden)
        self.assertEqual(PROBE._records, [])


class TestMemoryGaugeMatchesFrame(_ProbeCase):
    def _probe_bytes(self, multi_op):
        self.activate()
        block = _build(multi_op=multi_op)
        _run(block)
        self.assertEqual(len(PROBE._records), 1)
        return PROBE._records[0][4], block

    def _frame_bytes(self, multi_op, state):
        PROBE.deactivate()
        block = _build(multi_op=multi_op, recompute_modules=[_POINT])
        block.set_state_dict(state)
        self.assertIsInstance(block._rr, AutoRefinedRecompute)
        paddle.seed(_SEED)
        hidden = paddle.randn([_BATCH, _SEQ, _WIDE])
        retained = _retained_bytes(block, hidden)
        block._rr._frames.clear()
        return retained

    def test_single_op_boundary(self):
        measured, block = self._probe_bytes(multi_op=False)
        self.assertEqual(measured, self._frame_bytes(False, block.state_dict()))

    def test_multi_op_boundary_includes_intermediates(self):
        measured, block = self._probe_bytes(multi_op=True)
        retained = self._frame_bytes(True, block.state_dict())
        self.assertEqual(measured, retained)
        self.assertGreater(measured, _BATCH * _SEQ * _WIDE * 4)


class TestDisabledByDefault(_ProbeCase):
    def test_factory_unchanged_when_inactive(self):
        owner = _Block(_config())
        self.assertFalse(owner.use_rr)
        self.assertIsNone(owner._rr)

        listed = _Block(_config(recompute_modules=[_POINT]))
        self.assertTrue(listed.use_rr)
        self.assertIsInstance(listed._rr, AutoRefinedRecompute)

    def test_listed_point_is_priced_not_refined_when_active(self):
        self.activate()
        owner = _Block(_config(recompute_modules=[_POINT]))
        self.assertTrue(owner.use_rr)
        self.assertIsInstance(owner._rr, RRProbe)

    def test_unlisted_point_is_priced_when_active(self):
        self.activate()
        owner = _Block(_config())
        self.assertTrue(owner.use_rr)
        self.assertIsInstance(owner._rr, RRProbe)
        self.assertEqual(owner._rr.pending, 0)

    def test_preconditions_still_veto_when_active(self):
        self.activate()
        outside = _Block(_config(recompute_granularity=None))
        self.assertFalse(outside.use_rr)
        self.assertIsNone(outside._rr)

        virtual_pipeline = _Block(
            _config(virtual_pipeline_model_parallel_size=2)
        )
        self.assertFalse(virtual_pipeline.use_rr)
        self.assertIsNone(virtual_pipeline._rr)


class TestHandWrittenPoint(_ProbeCase):
    def _owner(self, **overrides):
        return types.SimpleNamespace(
            config=_config(**overrides), layer_number=3, is_mtp_layer=False
        )

    def test_returned_unchanged_when_inactive(self):
        function = paddle.tanh
        self.assertIs(
            PROBE.wrap(function, "flash_attn", self._owner()), function
        )

    def test_preconditions_veto(self):
        self.activate()
        function = paddle.tanh
        for overrides in (
            {"recompute_granularity": None},
            {"virtual_pipeline_model_parallel_size": 2},
        ):
            with self.subTest(**overrides):
                owner = self._owner(**overrides)
                self.assertIs(
                    PROBE.wrap(function, "flash_attn", owner), function
                )

    def test_only_recompute_pass_is_priced(self):
        self.activate()
        probed = PROBE.wrap(paddle.tanh, "flash_attn", self._owner())
        hidden = paddle.randn([_BATCH, _SEQ, _WIDE])
        hidden.stop_gradient = False
        recompute(probed, hidden).sum().backward()
        self.assertEqual(len(PROBE._records), 1)
        self.assertEqual(PROBE._records[0][0], ("flash_attn", 3, False))


if __name__ == "__main__":
    unittest.main()

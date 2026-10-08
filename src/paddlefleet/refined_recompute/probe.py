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
"""Measure the time and memory of every refined-recompute point, RR off.

Each point's boundary runs once in the recompute pass; that execution is what
RR would skip, so its span and allocated-memory delta are the point's price.
The time is an upper bound: it excludes RR's own cost and includes any wait on
another stream. Off unless ``RR_PROBE_DIR`` is set; when on, every point is
measured and none runs RR, whatever ``recompute_modules`` lists.

At exit each rank writes one JSON line per point, layer and step to
``rank{N}.jsonl``: ``{"point", "layer", "mtp", "n", "ms", "bytes"}``.
"""

import atexit
import json
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

__all__ = ["PROBE"]

_DIR_ENV = "RR_PROBE_DIR"


def _in_recompute_pass():
    from paddle.base import framework

    return framework._dygraph_tracer()._has_grad


class RRProbe:
    """Drop-in for ``AutoRefinedRecompute`` that measures instead of skipping."""

    def __init__(self, probe, point, layer, is_mtp):
        self._probe = probe
        self._key = (point, layer, bool(is_mtp))
        self.name = point

    @property
    def pending(self):
        return 0

    def __call__(self, function, *inputs):
        if not _in_recompute_pass():
            return function(*inputs)
        return self._probe.measure(self._key, function, inputs, {})


class _Probe:
    def __init__(self):
        self._directory = None
        self._records = []
        self._counts = defaultdict(int)
        self._exit_hooked = False

    def configure_from_env(self):
        directory = os.environ.get(_DIR_ENV)
        if directory:
            self.activate(directory)

    def activate(self, directory):
        self._directory = directory
        logger.warning(
            f"[RR-PROBE] measuring every refined-recompute point with RR off, "
            f"writing to {directory}"
        )

    def deactivate(self):
        self._directory = None
        self._records.clear()
        self._counts.clear()

    @property
    def active(self):
        return self._directory is not None

    def boundary(self, point, layer, is_mtp=False):
        return RRProbe(self, point, layer, is_mtp)

    def wrap(self, function, point, owner):
        """For points with a hand-written RR, which the probe keeps off."""
        if not self.active:
            return function
        from paddlefleet.recompute_utils import need_full_recompute

        vpp = getattr(
            owner.config, "virtual_pipeline_model_parallel_size", None
        )
        if not need_full_recompute(owner.layer_number, owner.config) or (
            vpp is not None and vpp > 1
        ):
            return function
        key = (
            point,
            owner.layer_number,
            bool(getattr(owner, "is_mtp_layer", False)),
        )

        def probed(*args, **kwargs):
            if not _in_recompute_pass():
                return function(*args, **kwargs)
            return self.measure(key, function, args, kwargs)

        return probed

    def measure(self, key, function, args, kwargs):
        import paddle

        before = paddle.device.cuda.memory_allocated()
        start = paddle.device.Event(enable_timing=True)
        end = paddle.device.Event(enable_timing=True)
        start.record()
        outputs = function(*args, **kwargs)
        end.record()
        # elapsed_time synchronizes, so spans are resolved in flush().
        delta = paddle.device.cuda.memory_allocated() - before
        occurrence = self._counts[key]
        self._counts[key] = occurrence + 1
        self._records.append((key, occurrence, start, end, delta))
        if not self._exit_hooked:
            # Registered after paddle is live so it runs before device teardown.
            atexit.register(self.flush)
            self._exit_hooked = True
        return outputs

    def flush(self):
        if not self._records:
            return
        import paddle

        paddle.device.synchronize()
        rank = os.environ.get("PADDLE_TRAINER_ID", "0")
        os.makedirs(self._directory, exist_ok=True)
        path = os.path.join(self._directory, f"rank{rank}.jsonl")
        records, self._records = self._records, []
        with open(path, "w") as stream:
            for key, occurrence, start, end, delta in records:
                point, layer, is_mtp = key
                stream.write(
                    json.dumps(
                        {
                            "point": point,
                            "layer": layer,
                            "mtp": is_mtp,
                            "n": occurrence,
                            "ms": start.elapsed_time(end),
                            "bytes": delta,
                        }
                    )
                    + "\n"
                )


PROBE = _Probe()
PROBE.configure_from_env()

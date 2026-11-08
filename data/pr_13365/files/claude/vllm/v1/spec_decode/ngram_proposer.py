# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np


class NgramProposer:

    def __init__(self):
        pass

    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
    ) -> Optional[np.ndarray]:
        """n-gram proposal via a vectorised sliding-window match.

        Same contract as the baseline KMP version: match the last ``n`` tokens
        against the earlier context and return the ``k`` tokens that follow the
        first occurrence, or ``None``. The scan is done with a single numpy
        windowed equality instead of a per-token Python loop, so there is no
        interpreter overhead per comparison and no JIT warm-up cost.
        """
        return self._find_subarray_np(context_token_ids, n, k)

    @staticmethod
    def _find_subarray_np(
        context_token_ids: np.ndarray,
        n: int,
        k: int,
    ) -> Optional[np.ndarray]:
        context_len = context_token_ids.shape[0]
        assert n > 0

        pattern = context_token_ids[-n:]
        # The baseline scans start index i in [0, context_len - n); a full match
        # of length n starting at s completes while the scan head stays in that
        # range, i.e. s in [0, context_len - 2n]. No earlier occurrence can exist
        # past that, so restrict the windows to that range.
        last_start = context_len - 2 * n
        if last_start < 0:
            return None

        # windows[s] = context[s : s + n] for s in [0, last_start].
        windows = np.lib.stride_tricks.sliding_window_view(
            context_token_ids[:last_start + n], n
        )
        matches = np.all(windows == pattern, axis=1)
        hit = np.argmax(matches)
        if not matches[hit]:
            return None

        # Baseline returns context[i : i + k] where i is the index just past the
        # matched pattern (start + n).
        i = int(hit) + n
        return context_token_ids[i:i + k]

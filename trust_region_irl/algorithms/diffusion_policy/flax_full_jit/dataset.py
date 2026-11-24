"""Fixed-length windows over the demonstrations, as the official SequenceSampler
(diffusion_policy/common/sampler.py: create_indices and sample_sequence).

Each window has `sequence_length` (= horizon) steps. Windows may start up to pad_before steps before an episode and
end up to pad_after steps after it; the missing steps repeat the episode's first / last step. Instead of copying
windows, gather_index() returns for every window the buffer row of each of its steps, so a batch is a single gather.
"""
import numpy as np


def create_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    """(buffer_start, buffer_end, sample_start, sample_end) per window; same loop as the official create_indices."""
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else int(episode_ends[i - 1])
        end_idx = int(episode_ends[i])
        episode_length = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            indices.append([buffer_start_idx, buffer_end_idx, start_offset, sequence_length - end_offset])
    return np.array(indices, dtype=np.int64)


def gather_index(indices, sequence_length):
    """(n_windows, sequence_length) buffer rows: before sample_start the first row, from sample_end on the last row."""
    buffer_start, buffer_end, sample_start, sample_end = (indices[:, i:i + 1] for i in range(4))
    j = np.arange(sequence_length)[None, :]
    rows = buffer_start + (j - sample_start)
    rows = np.where(j < sample_start, buffer_start, rows)
    rows = np.where(j >= sample_end, buffer_end - 1, rows)
    return rows.astype(np.int32)


def episode_ends_from(data):
    """Cumulative episode ends from episode_lengths, or from the absorbing / boundary marker on each last step."""
    if "episode_lengths" in data:
        return np.cumsum(np.asarray(data["episode_lengths"], dtype=np.int64))
    return np.where(np.asarray(data["absorbing"]).reshape(-1) > 0)[0] + 1

import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # 2nd seminar code
    batch_size = len(dataset_items)
    n_freqs = dataset_items[0]['spectrogram'].shape[1]
    texts = []
    text_lengths = torch.empty(batch_size, dtype=torch.int32)
    spec_lengths = torch.empty(batch_size, dtype=torch.int32)
    # audio_flag = True if 'audio' in dataset_items[0] else False
    # durations = torch.empty(batch_size)
    # audio_paths = []

    max_audio_len = 0

    for i, d in enumerate(dataset_items):
        spec_lengths[i] = d['spectrogram'].shape[2]
        text_lengths[i] = d['text_encoded'].shape[1]
        texts.append(d['text'])
        # durations[i] = d['duration']
        # audio_paths.append(d['audio_path'])
    # if audio_flag:
    #     for i, d in enumerate(dataset_items):
    #         max_audio_len = max(max_audio_len, d['audio'].shape[1])

    batch_specs = torch.zeros((batch_size, n_freqs, spec_lengths.max()))
    batch_texts = torch.zeros((batch_size, text_lengths.max()), dtype=torch.int32)

    for i, d in enumerate(dataset_items):
        batch_specs[i, :, :spec_lengths[i]] = d['spectrogram']
        batch_texts[i,    :text_lengths[i]] = d['text_encoded']

    result_batch = {
        "spectrogram": batch_specs,
        "text": texts,
        "text_encoded": batch_texts,
        'spectogram_length' : spec_lengths,
        'text_encoded_length' : text_lengths,
        # "duration": durations,
        # "audio_path": audio_paths,
    }

    # if audio_flag:
    #     batch_audio = torch.zeros((batch_size, max_audio_len))
    #     for i, d in enumerate(dataset_items):
    #         audio = d['audio']
    #         batch_audio[i, :audio.shape[1]] = audio
    #     result_batch['audio'] = batch_audio
    return result_batch

from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    # DeepSpeech (V1) model with added BatchNorm
    def __init__(self, n_feats, n_class, fc_hidden=2048, context=9, stride=2, p=0.05, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.conv = nn.Conv1d(n_feats, n_feats, kernel_size=2*context + 1, stride=stride, groups=n_feats)
        self.tail = Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Dropout(p),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.Dropout(p),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.Dropout(p),
            nn.Hardtanh(0, 20, inplace=True)
        )
        self.rnn = nn.RNN(
            input_size=fc_hidden, hidden_size=fc_hidden, bidirectional=True, nonlinearity='relu', batch_first=True
        )

        self.head = Sequential(
            nn.Hardtanh(0, 20, inplace=True),
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        b_size = spectrogram.shape[0]
        lengths = self.transform_input_lengths(spectrogram_length)

        out = self.conv(spectrogram)
        out = out.transpose(1, 2)                       # [B, F, T] -> # [B, T, F]
        out = self.tail(out.reshape(-1, out.shape[2]))  # [B, T, F] -> [B * T, F] for BatchNorm1d
        out = out.view(b_size, -1, out.shape[1])        # [B * T, F] -> [B, T, F]
        packed = nn.utils.rnn.pack_padded_sequence(out, lengths, enforce_sorted=False, batch_first=True)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out,)
        out = out.view(out.shape[0], out.shape[1], out.shape[2]//2, 2).sum(dim=-1)  # sum both RNNs
        return {"logits": self.head(out.transpose(0, 1))}

    def transform_input_lengths(self, input_lengths):
        return ((input_lengths - self.conv.kernel_size[0] - 2) // self.conv.stride[0]) + 1

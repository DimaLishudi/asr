from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    # DeepSpeech (V1) model with added BatchNorm
    def __init__(self, n_feats, n_class, fc_hidden=2048, context=9, stride=2, p=0.05, depthwise=False, **batch):
        super().__init__(n_feats, n_class, **batch)
        if depthwise:
            self.conv = nn.Conv1d(n_feats, n_feats, kernel_size=2*context + 1, stride=stride, groups=n_feats)
        else:
            self.conv = nn.Conv1d(n_feats, n_feats, kernel_size=2*context + 1, stride=stride)
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

        out = self.conv(spectrogram.squeeze())
        out = out.transpose(1, 2)                       # [B, F, T] -> # [B, T, F]
        out = self.tail(out.reshape(-1, out.shape[2]))  # [B, T, F] -> [B * T, F] for BatchNorm1d
        out = out.view(b_size, -1, out.shape[1])        # [B * T, F] -> [B, T, F]
        packed = nn.utils.rnn.pack_padded_sequence(out, lengths, enforce_sorted=False, batch_first=True)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.view(out.shape[0], out.shape[1], out.shape[2]//2, 2).sum(dim=-1)  # sum both RNNs
        return {"logits": self.head(out)}

    def transform_input_lengths(self, input_lengths):
        return ((input_lengths - self.conv.kernel_size[0] - 2) // self.conv.stride[0]) + 1


class DeepSpeech2(BaseModel):
    # DeepSpeech2 with 1 Conv2d and multiple bidirectional rnns (no lookahead)
    def __init__(self, n_feats, n_class, fc_hidden=1024, num_rnn=3, stride=2, p=0.05, **batch):
        super().__init__(n_feats, n_class, **batch)
        F_C = 20  # feature context
        T_C = 5   # time context
        C = 32    # channels

        self.num_rnn = num_rnn
        self.conv = nn.Conv2d(1, C, kernel_size=(2*F_C + 1,  2*T_C + 1), stride=stride, padding=(F_C, T_C))

        rnns = [nn.RNN(
                input_size=n_feats*C//stride, hidden_size=fc_hidden, bidirectional=True, batch_first=True
            )]

        bns = [nn.BatchNorm1d(C*n_feats//stride)]

        for i in range(num_rnn-1):
            rnns.append(nn.RNN(
                input_size=fc_hidden, hidden_size=fc_hidden, bidirectional=True,  batch_first=True
            ))
            bns.append(nn.BatchNorm1d(fc_hidden))

        self.rnns = nn.ModuleList(rnns)
        self.bns  = nn.ModuleList(bns)

        self.head = Sequential(
            # nn.BatchNorm1d(fc_hidden),
            # nn.Dropout(p),
            # nn.Hardtanh(0, 20, inplace=True),
            # nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.Dropout(p),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        b, f, t = spectrogram.shape
        lengths = self.transform_input_lengths(spectrogram_length)

        out = self.conv(spectrogram.view(b, 1, f, t))
        out = out.view(b, -1, out.shape[3]).transpose(1, 2)  # [B, C, F, T] -> # [B, T, C*F]
        for i in range(self.num_rnn):
            out = out.reshape(-1, out.shape[2])  # [B, T, F] -> [B * T, F] for BatchNorm1d
            out = self.bns[i](out)
            out = out.view(b, -1, out.shape[1])  # [B * T, F] -> [B, T, F]
            packed = nn.utils.rnn.pack_padded_sequence(out, lengths, enforce_sorted=False, batch_first=True)
            out, _ = self.rnns[i](packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            out = out.view(out.shape[0], out.shape[1], out.shape[2]//2, 2).sum(dim=-1)  # sum both RNNs
        out = self.head(out.reshape(-1, out.shape[2]))
        return {"logits": out.view(b, -1, out.shape[1])}

    def transform_input_lengths(self, input_lengths):
        return ((input_lengths - 1) // self.conv.stride[0]) + 1

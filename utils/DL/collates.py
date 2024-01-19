
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def padding_x(batch):
    x, y = zip(*batch)
    x = [i.permute(1, 0) for i in x]
    # Pad sequences if needed (inputs L-C)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)

    if isinstance(y, tuple) and y[0].shape[0] >= 2:
        y = torch.stack(y, dim=0)
        # y = torch.concat((y[0].unsqueeze(0), y[1].unsqueeze(0)), dim=0)
    else:
        y = torch.LongTensor(y)

    return x_padded.permute(0, 2, 1), y


def keep_unchanged(batch):
    x, y = zip(*batch)

    return x[0], y[0]
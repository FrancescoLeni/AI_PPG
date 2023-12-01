
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def padding_x(batch):
    print(type(batch))
    x, y = zip(*batch)
    x = [i.permute(1,0) for i in x]
    # Pad sequences if needed
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)

    return x_padded.permute(0,2,1), y
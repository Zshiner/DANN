import torch
from transformers import BertModel
import pathlib

class CustomBert(torch.nn.Module):
    def __init__(self, output_dim, drop_rate, bert_dim, *args, **kwargs):
        super(CustomBert, self).__init__()
        self.bert = None

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(bert_dim, 968),
            torch.nn.ReLU(),
            torch.nn.Linear(968, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),
        )

    def forward(self, batch):
        x, mask = batch
        x = x.to('cuda')
        mask = mask.to('cuda')


        pooler_output = self.bert(x, mask)['pooler_output']
        out = self.classifier(pooler_output)

        return out



class TCMBert(CustomBert):
    def __init__(self, output_dim,*args, **kwargs):
        super(TCMBert, self).__init__(output_dim,*args, **kwargs)
        self.bert = BertModel.from_pretrained(
            pathlib.Path(__file__).parent / 'tcmbert'
        )


class ZYBert(CustomBert):
    def __init__(self, output_dim,*args, **kwargs):
        super(ZYBert, self).__init__(output_dim,*args, **kwargs)
        self.bert = BertModel.from_pretrained(
            pathlib.Path(__file__).parent / 'zybert'
        )
import torch
from transformers import BertModel
import pathlib

class CustomBertCNN(torch.nn.Module):
    def __init__(self, output_dim, drop_rate, bert_dim, filter_sizes=[2, 3, 4], num_filters=128, *args, **kwargs):
        super(CustomBertCNN, self).__init__()
        self.bert = None

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(
                in_channels=bert_dim,  # BERT隐藏层维度
                out_channels=num_filters,  # 每个卷积核的输出通道数
                kernel_size=fs,  # 卷积核尺寸
                padding=(fs - 1) // 2  # 保持序列长度不变
            )
            for fs in filter_sizes
        ])

        cnn_out_dim = len(filter_sizes) * num_filters

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(cnn_out_dim, 968),
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


        bert_output = self.bert(x, mask).last_hidden_state

        conv_input = bert_output.permute(0, 2, 1)
        conv_outputs = []
        for conv in self.convs:
            # 卷积: [batch, num_filters, seq_len]
            out = torch.relu(conv(conv_input))
            # 全局最大池化: [batch, num_filters]
            pooled = torch.max(out, dim=2)[0]
            conv_outputs.append(pooled)

        # 拼接所有卷积层的输出
        cnn_features = torch.cat(conv_outputs, dim=1)

        out = self.classifier(cnn_features)

        return out


class BertCNN(CustomBertCNN):
    def __init__(self, output_dim,*args, **kwargs):
        super(BertCNN, self).__init__(output_dim,*args, **kwargs)
        self.bert = BertModel.from_pretrained(
            pathlib.Path(__file__).parent / 'bert-base-chinese'
        )
# FT-Transformer & TabTransformer 表格数据基线模型
# 兼容 Comparator 框架 (scripts/deep.py)
#
# 用法:
#   python tabel.py --dataset sub                     # 运行全部
#   CUDA_VISIBLE_DEVICES=0 python tabel.py --only FT-Transformer --dataset sub
#   CUDA_VISIBLE_DEVICES=0 python tabel.py --only TabTransformer --s 1

# CUDA_VISIBLE_DEVICES=3 python tabel.py --only TabTransformer --s 4

from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import argparse
import math
import torch
import torch.nn as nn

from config.config import Config
from scripts.get_data import gd
from scripts.deep import Comparator


# ======================================================================
#  FT-Transformer (Gorishniy et al., NeurIPS 2021)
#  Revisiting Deep Learning Models for Tabular Data
#
#  核心思想: 每个数值特征独立投影为 token，加 [CLS] token，
#  经 Transformer Encoder 建模特征交互，取 [CLS] 输出分类。
# ======================================================================

class TabFTTransformer(nn.Module):

    def __init__(self, input_dim, output_dim, embedding_dim=64,
                 num_heads=4, num_blocks=3,
                 attn_dropout=0.1, ff_dropout=0.1, **kwargs):
        super().__init__()
        d = embedding_dim

        # Feature Tokenizer: 每个特征有独立的 W_j (d,) 和 b_j (d,)
        # token_j = x_j * W_j + b_j
        self.W = nn.Parameter(torch.empty(input_dim, d))
        self.b = nn.Parameter(torch.empty(input_dim, d))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.zeros_(self.b)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Transformer Encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=ff_dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, output_dim),
        )

    def forward(self, x):
        # x: (B, F)
        # 每特征独立投影: (B, F, 1) * (1, F, d) + (1, F, d) → (B, F, d)
        tokens = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)

        # 拼接 [CLS]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, F+1, d)

        out = self.encoder(tokens)    # (B, F+1, d)
        return self.head(out[:, 0])   # [CLS] → (B, output_dim)


# ======================================================================
#  TabTransformer (Huang et al., 2020)
#  TabTransformer: Tabular Data Modeling Using Contextual Embeddings
#
#  适配全连续特征: 每特征独立投影 + column embedding → Transformer → [CLS]
#
#  与 FT-Transformer 的架构区分:
#    FT-Transformer: 纯特征投影 (无 column embed, 无 input norm) + post-norm
#    TabTransformer:  特征投影 + column embedding + input LayerNorm + pre-norm
# ======================================================================

class TabTabTransformer(nn.Module):

    def __init__(self, input_dim, output_dim, embedding_dim=64,
                 num_heads=4, num_blocks=3,
                 ff_dropout=0.1, mlp_hidden=None, **kwargs):
        super().__init__()
        d = embedding_dim
        if mlp_hidden is None:
            mlp_hidden = d * 2

        # 每特征独立投影
        self.W = nn.Parameter(torch.empty(input_dim, d))
        self.b = nn.Parameter(torch.empty(input_dim, d))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.zeros_(self.b)

        # Column embedding: 数据无关的列位置信号 (TabTransformer 核心组件)
        self.col_embed = nn.Parameter(torch.empty(1, input_dim, d))
        nn.init.normal_(self.col_embed, std=d ** -0.5)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)

        # 输入 LayerNorm — 稳定 token 分布
        self.input_norm = nn.LayerNorm(d)

        # Pre-norm Transformer (Xiong et al. 2020)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=ff_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)

        # MLP Head
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, mlp_hidden),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)
        # 每特征独立投影 + column embedding → LayerNorm
        tokens = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        tokens = tokens + self.col_embed
        tokens = self.input_norm(tokens)
        # 拼接 [CLS] (在 LayerNorm 之后，CLS 不需要 column embedding)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)      # (B, F+1, d)
        out = self.encoder(tokens)                     # (B, F+1, d)
        return self.head(out[:, 0])                    # [CLS] → (B, output_dim)


# ======================================================================
#  Comparator 子类: 参数搜索网格
# ======================================================================

class TabComparator(Comparator):

    def get_model_param_grid(self):
        input_dim = self.data[0][0].shape[-1]
        output_dim = self.data[0][1].shape[-1]

        grid = {}

        grid['FT-Transformer'] = (TabFTTransformer, {
            'input_dim': [input_dim],
            'output_dim': [output_dim],
            'embedding_dim': [256],
            'num_heads': [16],
            'num_blocks': [4],
            'attn_dropout': [0.1],
            'ff_dropout': [0.1],
            'lr': [1e-4],
            'batch_size': [256],
            'tor': [5],
        })

        # grid['TabTransformer'] = (TabTabTransformer, {
        #     'input_dim': [input_dim],
        #     'output_dim': [output_dim],
        #     'embedding_dim': [64,32,16,128,512],
        #     'num_heads': [8,4,16],
        #     'num_blocks': [4],
        #     'ff_dropout': [0.1],
        #     'lr': [5e-4,1e-4,2e-3],
        #     'batch_size': [128],
        #     'tor': [5],
        # })
        grid['TabTransformer'] = (TabTabTransformer, {
            'input_dim': [input_dim],
            'output_dim': [output_dim],
            'embedding_dim': [512],
            'num_heads': [16],
            'num_blocks': [4],
            'ff_dropout': [0.1],
            'lr': [1e-4],
            'batch_size': [128],
            'tor': [10],
        })


        return grid


# ======================================================================
#  入口
# ======================================================================
# CUDA_VISIBLE_DEVICES=0 python optimization_bert_sub_sub.py --only TabTransformer --s 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FT-Transformer / TabTransformer 对比实验')
    parser.add_argument('--only', type=str, default=None,
                        help='仅运行指定模型 (FT-Transformer / TabTransformer)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='指定数据集名称')
    parser.add_argument('--s', type=str, default=None,
                        help='指定并发名称')
    args = parser.parse_args()


    for (dataset_name, split_list) in gd.get_all():

        if 'sub' in dataset_name:
            task_type = 'multi_class'
        elif dataset_name == 'lung':
            task_type = 'multi_label'
        else:
            raise UserWarning("任务类型未指定")

        print(dataset_name)
        if '15' in dataset_name or 'repeat_0' in dataset_name:
            continue

        STOP_mark = False
        for i in ['sub_14_repeat_9']:
            if i not in dataset_name:
                STOP_mark = True
        if STOP_mark:
            continue


        start_mark = False

        # 并发实验，用启动参数分发任务
        if args.s == '1':
            for i in ['_repeat_9','_repeat_5']:
                if i in dataset_name:
                    start_mark = True
        elif args.s == '2':
            for i in ['_repeat_6','_repeat_7']:
                if i in dataset_name:
                    start_mark = True
        elif args.s == '3':
            for i in ['_repeat_8','_repeat_9']:
                if i in dataset_name:
                    start_mark = True
        elif args.s == '4':
            for i in ['_repeat_7','_repeat_8']:
                if i in dataset_name:
                    start_mark = True
        else:
            print('并发指定错误，请检查')
            exit()

        if not start_mark:
            continue

        print('=' * 60)
        print(f'开始实验: {dataset_name} (任务类型: {task_type})')
        print('=' * 60)

        cp = TabComparator(
            split_list, dataset_name,
            save_dir=f'./out/{dataset_name}',
            seed=Config.seed,
            default_args=False,
            device='cuda',
            type=task_type,
            top_k=1 if task_type == 'multi_class' else 2,
            only_model=args.only,
            process_num=1,
            save_model=False,
            tor=10,
        )
        cp.run()

    print('所有实验完成。')

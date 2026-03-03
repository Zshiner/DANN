import torch
import numpy as np

def totokenid_bert(tokenizer, sentence: list, max_len: int, tags=None, tags_map=None):
    if len(sentence) >= max_len - 2:
        sentence = sentence[:max_len - 2]

        if not isinstance(tags, type(None)):
            tags = tags[:max_len - 2]

    #
    sentence = tokenizer.tokenize(sentence[0])
    sentence = ["[CLS]"] + sentence + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(sentence)

    # bert默认提供的mask中，原数据为1，padding部分为0，这里与其保持一致
    attention_mask = [1 for i in input_ids]

    # tokenizer.encode 自动为sentence添加了101，102两个特殊字符，因此tags也需要相应补充
    if not isinstance(tags, type(None)):
        tags = [tags_map['O']] + tags + [tags_map['O']]

    return input_ids, attention_mask, tags


def padbatch2tokenid(batch):
    """
    如果按照最大长度提前padding所有数据，会造成padding过多，容易引发梯度消失且消耗存储空间。
    因此，在dataloader取出batch_data后，基于dataloader的collate_fn=padbatch2tokenid（本函数），、
    可以对每一个单独的batch_data进行padding，此时padding的长度就是当前batch_data的最大长度，而不是全局最大长度
    :param batch: [inputids, mask_attention, tags] or [inputids, mask_attention]
    :return: batch
    """

    maxlen = max([len(i[0][0]) for i in batch])
    padding_list = [0 for i in range(maxlen)]

    #
    input_ids_pad = [list(i[0][0]+padding_list)[0:maxlen] for i in batch]
    mask_pad = [list(i[0][1] + padding_list)[0:maxlen] for i in batch]
    y = np.array([i[1].detach().cpu().numpy() for i in batch])


    return (torch.tensor(input_ids_pad), torch.tensor(mask_pad)), torch.tensor(y)
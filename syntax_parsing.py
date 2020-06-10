#! -*- coding: utf-8 -*-
# BERT无监督提取句法结构
# 介绍：https://kexue.fm/archives/7476

import json
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import uniout
import jieba

# BERT配置
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# 文本编码
text = u'计算机的鼠标有什么比较特殊的用途呢'
words = jieba.lcut(text)
spans = []
token_ids = [tokenizer._token_start_id]
for w in words:
    w_ids = tokenizer.encode(w)[0][1:-1]
    token_ids.extend(w_ids)
    spans.append((len(token_ids) - len(w_ids), len(token_ids)))

token_ids.append(tokenizer._token_end_id)
length = len(spans)


def dist(x, y):
    """距离函数（默认用欧氏距离）
    可以尝试换用内积或者cos距离，结果差不多。
    """
    return np.sqrt(((x - y)**2).sum())


batch_token_ids = np.array([token_ids] * (length * (length + 1) / 2))
batch_segment_ids = np.zeros_like(batch_token_ids)
k, mapping = 0, {}
for i in range(length):
    for j in range(i, length):
        mapping[i, j] = k
        batch_token_ids[k, spans[i][0]:spans[i][1]] = tokenizer._token_mask_id
        batch_token_ids[k, spans[j][0]:spans[j][1]] = tokenizer._token_mask_id
        k += 1

vectors = model.predict([batch_token_ids, batch_segment_ids])
distances = np.zeros((length, length))

for i in range(length):
    for j in range(i + 1, length):
        vi = vectors[mapping[i, i], spans[i][0]:spans[i][1]].mean(0)
        vij = vectors[mapping[i, j], spans[i][0]:spans[i][1]].mean(0)
        distances[i, j] = dist(vi, vij)
        vj = vectors[mapping[j, j], spans[j][0]:spans[j][1]].mean(0)
        vji = vectors[mapping[i, j], spans[j][0]:spans[j][1]].mean(0)
        distances[j, i] = dist(vj, vji)


def build_tree(words, distances):
    """递归解析句子层次结构
    """
    if len(words) == 1:
        return [words[0]]
    elif len(words) == 2:
        return [words[0], [words[1]]]
    else:
        k = np.argmax([
            distances[:i, :i].mean() + distances[i:, i:].mean() -
            distances[:i, i:].mean() - distances[i:, :i].mean() \
            for i in range(1, len(words) - 1)
        ]) + 1
        return [
            build_tree(words[:k], distances[:k, :k]),
            [words[k],
             build_tree(words[k + 1:], distances[k + 1:, k + 1:])]
        ]


# 用json.dumps做简单的可视化
json.dumps(build_tree(words, distances), indent=4, ensure_ascii=False)
"""输出：
[
    [
        [
            "计算机"
        ],
        [
            "的",
            [
                "鼠标"
            ]
        ]
    ],
    [
        "有",
        [
            [
                [
                    "什么"
                ],
                [
                    "比较",
                    [
                        "特殊"
                    ]
                ]
            ],
            [
                "的",
                [
                    "用途",
                    [
                        "呢"
                    ]
                ]
            ]
        ]
    ]
]
"""

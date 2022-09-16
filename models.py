from torch import nn
from torch.nn import functional as F
import torch 

class TextCNN(nn.Module):
    def __init__(self, vocab, embedding = None, kernel_size = [3, 4, 5], filter_num = 16, dropout=0.2, **kwargs):
        """
        - param:
            - embedding: if "pre" -> pre-train,wiki_word2vec_50, else int will be set as embedding_dim
        """
        super(TextCNN, self).__init__()
        
        if isinstance(embedding, int):
            self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding, padding_idx=vocab.vocab["<pad>"])
            self.d_m = embedding
        if embedding is None:  # 自定义传入embedding
            self.embedding = None
            self.d_m = len(vocab)
            pass
            
        self.convlution_block = nn.ModuleList([nn.Conv2d(1, filter_num, kernel_size=(i, self.d_m), padding="valid", ) for i in kernel_size ])
        
        self.linear = nn.Linear(filter_num * len(kernel_size), 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        tmp_device = x.device  # 由于有的时候device会改变，因为预训练的词向量没有加入到device中
        if self.embedding:
            tmp_device = x.device  # 由于有的时候device会改变，因为预训练的词向量没有加入到device中
            x = self.embedding(x)  #[bn, seqlen] -> [bn, seqlenm d_m]
            x = x.to(tmp_device)
        
        x = x.unsqueeze(1)  # [bn, seqlen, d_m] -> [bn, 1, seqlen, d_m]
        out = []
        for conv in self.convlution_block:
            x_conv = conv(x)
            # print(x.device, x_conv.device)
            x_conv = F.relu(x_conv)
            x_conv = nn.MaxPool2d(x_conv.shape[-2:])(x_conv)  # 使用卷积后的最后两个维度的大小作为池化的大小：[bn, channel, seq_len ,1] ->[bn, channel, 1, 1]，可以修改为avgpool
            out.append(x_conv.flatten(1))  # [bn, channel, 1, 1] -> [bn, channel]
        
        out = torch.cat(out, dim = 1)
        out = self.dropout(out)
        return torch.sigmoid(self.linear(out))
        # return torch.sigmoid(self.linear(out))
        

        
    
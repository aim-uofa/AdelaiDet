import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from adet.layers import conv_with_kaiming_uniform


class BidirectionalLSTM(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CRNN, self).__init__()
        conv_func = conv_with_kaiming_uniform(norm="GN", activation=True)
        convs = []
        for i in range(2):
            convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.convs = nn.Sequential(*convs)
        self.rnn = BidirectionalLSTM(in_channels, in_channels, in_channels)

    def forward(self, x):
        # average along H dimension
        x = self.convs(x)
        x = x.mean(dim=2)  # NxCxW
        x = x.permute(2, 0, 1)  # WxNxC
        x = self.rnn(x)
        return x


# apply attention
class Attention(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Attention, self).__init__()
        self.hidden_size = in_channels
        self.output_size = cfg.MODEL.BATEXT.VOC_SIZE + 1
        self.dropout_p = 0.1
        self.max_len = cfg.MODEL.BATEXT.NUM_CHARS

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # test
        self.vat = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        '''
        hidden: 1 x n x self.hidden_size
        encoder_outputs: time_step x n x self.hidden_size (T,N,C)
        '''
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # test
        batch_size = encoder_outputs.shape[1]

        alpha = hidden + encoder_outputs
        alpha = alpha.view(-1, alpha.shape[-1]) # (T * n, hidden_size)
        attn_weights = self.vat(torch.tanh(alpha))  # (T * n, 1)
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2,1,0)) # (T, 1, n)  -> (n, 1, T)
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute((1, 0, 2)))

        if embedded.dim() == 1:
            embedded = embedded.unsqueeze(0)
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0) # (1, n, hidden_size)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden) # (1, n, hidden_size)

        output = F.log_softmax(self.out(output[0]), dim=1)  # (n, hidden_size)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result

    def prepare_targets(self, targets):
        target_lengths = (targets != self.output_size - 1).long().sum(dim=-1)
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        return target_lengths, sum_targets


class ATTPredictor(nn.Module):
    def __init__(self, cfg):
        super(ATTPredictor, self).__init__()
        in_channels = cfg.MODEL.BATEXT.CONV_DIM
        self.CRNN = CRNN(cfg, in_channels)
        self.criterion = torch.nn.NLLLoss()
        self.attention = Attention(cfg, in_channels)
        self.teach_prob = 0.5

    def forward(self, rois, targets=None):
        rois = self.CRNN(rois)
        if self.training:
            target_variable = targets
            _init = torch.zeros((rois.size()[1], 1)).long()
            _init = torch.LongTensor(_init).to(rois.device)
            target_variable = torch.cat((_init, target_variable.long()), 1)
            target_variable = target_variable.to(rois.device)
            decoder_input = target_variable[:,0] # init decoder, from 0
            decoder_hidden = self.attention.initHidden(rois.size()[1]).to(rois.device) # batch rois.size[1]
            loss = 0.0

            for di in range(1, target_variable.shape[1]):
                decoder_output, decoder_hidden, decoder_attention = self.attention( #  decoder_output (nbatch, ncls)
                    decoder_input, decoder_hidden, rois)
                loss += self.criterion(decoder_output, target_variable[:,di])
                teach_forcing = True if random.random() > self.teach_prob else False
                if teach_forcing:
                    decoder_input = target_variable[:,di]  # Teacher forcing
                else:
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.squeeze()
                    decoder_input = ni

            return None, loss
        else:
            n = rois.size()[1]
            decodes = torch.zeros((n, self.attention.max_len))
            prob = 1.0
            decoder_input = torch.zeros(n).long().to(rois.device) 
            decoder_hidden = self.attention.initHidden(n).to(rois.device)
            for di in range(self.attention.max_len):
                decoder_output, decoder_hidden, decoder_attention = self.attention(
                    decoder_input, decoder_hidden, rois)
                probs = torch.exp(decoder_output)
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni
                prob *= probs[:, ni]
                decodes[:, di] = decoder_input
            return decodes, None
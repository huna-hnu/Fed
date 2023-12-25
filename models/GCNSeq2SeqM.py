import numpy as np
import torch
import torch.nn as nn

from models.fedsageplus import GNN, GCN


class GRUSeq2SeqWithGraphNet(nn.Module):
    def __init__(self, input_size, output_size, args):
        super(GRUSeq2SeqWithGraphNet,self).__init__()
        self.args = args
        input_size = input_size
        hidden_size = args.hidden_size
        output_size = output_size
        dropout = args.dropout_t
        gru_num_layers = args.gru_num_layers
        self.use_curriculum_learning = args.use_curriculum_learning
       # self.cl_decay_steps = args.cl_decay_steps

        if args.use_gcn_m:

            self.encoder = nn.GRU(
                input_size , hidden_size, num_layers=gru_num_layers, dropout=dropout
            )

            self.decoder = nn.GRU(
                input_size, 2*hidden_size, num_layers=gru_num_layers, dropout=dropout
            )

            self.out_net = nn.Linear(hidden_size*2, output_size)
        elif args.use_gcn_t:

            self.gcn = GCN(nfeat=hidden_size,  # 提取修补图中的空间特征 两层GCN加中间的dropout层
                nhid=args.hidden,  # 中间隐藏维度
                nout=hidden_size,
                dropout=args.dropout, device=args.device)

            self.encoder = nn.GRU(
                input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
            )

            self.decoder = nn.GRU(
                input_size, 2 * hidden_size, num_layers=gru_num_layers, dropout=dropout
            )

            self.out_net = nn.Linear(hidden_size * 2, output_size)

        else:
            self.encoder = nn.GRU(
                input_size , hidden_size, num_layers=gru_num_layers, dropout=dropout
            )

            self.decoder = nn.GRU(
                input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
            )

            self.out_net = nn.Linear(hidden_size, output_size)






    def _format_input_data(self, data): # B, T, N, F
        x,  y= data['x'],  data['y']

        batch_num, node_num = x.shape[0], x.shape[2]
        return x, y, batch_num, node_num

    def forward(self, data, batches_seen, mendGcn_encoding, return_encoding=False):
        h_encode = self.forward_encoder(data)
        return self.forward_decoder(data, h_encode, batches_seen,mendGcn_encoding)

    def forward_encoder(self, data): # B, T, N, F
        x,  y, batch_num, node_num = self._format_input_data(data)
        x_input = x.permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F

        out, h_encode = self.encoder(x_input)
        return h_encode #  # layers x (B x N) x hidden_dim

    def forward_decoder(self, data, h_encode, batches_seen, mendGcn_encoding, return_encoding=False, ):
        x,  y,  batch_num, node_num = self._format_input_data(data)

        x_input = x.permute(1, 0, 2, 3).flatten(1, 2)  # T， B*N， F

        encoder_h = h_encode # num_layer, B*N, hidden_dim

        if self.args.use_gcn_t:
            adj = data['adj']

            graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(1, 0, 2,
                                                                                                          3)  #   B x L x N x F
            graph_encoding = self.gcn(graph_encoding, adj) # B x L x N x  F

            graph_encoding = graph_encoding.permute(1, 0, 2, 3).flatten(1, 2)  # L x (B x N) x F


            h_encode = torch.cat([h_encode, graph_encoding], dim=-1)

        if self.args.use_gcn_m:
            h_encode = torch.cat([h_encode, mendGcn_encoding], dim=-1) # layers, B*N, hidden_dim*2 默认中间使用MendGCN

        if self.training and (not self.use_curriculum_learning):
            y_input = y.permute(1, 0, 2, 3).flatten(1, 2) # T, B*N, 1

            y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0)  # T, B*N, F

            out_hidden, _ = self.decoder(y_input, h_encode) # T ,, B*N, F
            out = self.out_net(out_hidden)
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3) # B, T, N, 1
        else:
            last_input = x_input[-1:] # 1, B*N, 1
            last_hidden = h_encode
            step_num = y.shape[1]   # 预测时间步 B, T, N, 1
            out_steps = []
            y_input = y.permute(1, 0, 2, 3).flatten(1, 2) # T, B*N, 1
          #  y_attr_input = y_attr.permute(1, 0, 2, 3).flatten(1, 2)
            for t in range(step_num):
                out_hidden, last_hidden = self.decoder(last_input, last_hidden)
                out = self.out_net(out_hidden) # 1 x (B x N) x 1
                out_steps.append(out)

                last_input_from_output = out
                last_input_from_gt = y_input[t:t + 1]  # 1, B*N,1
                if self.training:
                    p_gt = self._compute_sampling_threshold(batches_seen)
                    p = torch.rand(1).item()
                    if p <= p_gt:
                        last_input = last_input_from_gt
                    else:
                        last_input = last_input_from_output
                else:
                    last_input = last_input_from_output
            out = torch.cat(out_steps, dim=0) # T, B*N, 1
            out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3) # B, T, N,1
        # if type(data) is Batch:
        #     out = out.squeeze(0).permute(1, 0, 2) # N x T x 1
        if return_encoding:
            return out, encoder_h
        else:
            return out #  # B, T, N,1


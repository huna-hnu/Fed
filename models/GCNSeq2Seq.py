import numpy as np
import torch
import torch.nn as nn


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
        self.cl_decay_steps = args.cl_decay_steps

        if args.use_gcn:
            self.encoder = nn.GRU(
                input_size * 2, hidden_size, num_layers=gru_num_layers, dropout=dropout
            )

            self.decoder = nn.GRU(
                input_size*2, hidden_size, num_layers=gru_num_layers, dropout=dropout
            )
        else:
            self.encoder = nn.GRU(
                input_size , hidden_size, num_layers=gru_num_layers, dropout=dropout
            )

            self.decoder = nn.GRU(
                input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
            )


        self.out_net = nn.Linear(hidden_size , output_size)

    def _format_input_data(self, data): # B, T, N, F
        x,  y= data['x'],  data['y']

        batch_num, node_num = x.shape[0], x.shape[2]
        return x, y, batch_num, node_num

    def forward(self, data, batches_seen, mendGcn_encoding, return_encoding=False):
        h_encode = self.forward_encoder(data, mendGcn_encoding)
        return self.forward_decoder(data, h_encode, batches_seen,mendGcn_encoding)

    def forward_encoder(self, data, mendGcn_encoding): # B, T, N, F
        x,  y, batch_num, node_num = self._format_input_data(data)
        x_input = x.permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F

        if self.args.use_gcn:
            x_input = torch.cat([x_input, mendGcn_encoding], dim=-1)  # T x (B x N) x 2F

        else:
            x_input = x_input  # T x (B x N) x 2F

        out, h_encode = self.encoder(x_input)
        return h_encode #  # layers x (B x N) x hidden_dim

    def forward_decoder(self, data, h_encode, batches_seen, mendGcn_encoding, return_encoding=False ):
        x,  y,  batch_num, node_num = self._format_input_data(data)

        x_input = x.permute(1, 0, 2, 3).flatten(1, 2)  # T， B*N， F
        if self.args.use_gcn:
            x_input = torch.cat([x_input, mendGcn_encoding], dim=-1)  # T, B*N, 2*F
        encoder_h = h_encode # num_layer, B*N, hidden_dim

       # h_encode = torch.cat([h_encode, mendGcn_encoding], dim=-1) # layers, B*N, hidden_dim*2

        if self.training and (not self.use_curriculum_learning):
            y_input = y.permute(1, 0, 2, 3).flatten(1, 2) # T, B*N, 1

            if self.args.use_gcn:
                y_input = torch.cat([y_input, mendGcn_encoding], dim=-1) # T, B*N, 2*F

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
                  #
                if self.args.use_gcn:
                    last_input_from_output = torch.cat([out, mendGcn_encoding[t:t+1]], dim=-1)
                    last_input_from_gt = torch.cat([y_input[t:t+1], mendGcn_encoding[t:t+1]], dim=-1) # 1, B*N,1
                else:
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


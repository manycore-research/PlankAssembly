# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (TransformerDecoder, TransformerDecoderLayer,
                      TransformerEncoder, TransformerEncoderLayer)


class PlankModel(nn.Module):

    def __init__(self,
                 num_model=512,
                 num_head=8,
                 num_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_view=3,
                 num_type=2,
                 num_input_dof=4,
                 num_output_dof=6,
                 max_input_length=400,
                 max_output_length=128,
                 vocab_size=514,
                 token=None):
        super(PlankModel, self).__init__()

        max_num_input = math.ceil(max_input_length / num_input_dof)
        max_num_output = math.ceil(max_output_length / num_output_dof)

        self.num_model = num_model
        self.max_num_input = max_num_input
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_input_dof = num_input_dof
        self.num_output_dof = num_output_dof
        self.max_num_output = max_num_output
        self.vocab_size = vocab_size

        self.token = token

        # input sequence
        self.input_embeddings = nn.ModuleDict({
            'input_value': nn.Embedding(vocab_size, num_model),
            'input_pos': nn.Embedding(max_num_input, num_model),
            'input_coord': nn.Embedding(num_input_dof, num_model),
            'input_view': nn.Embedding(num_view, num_model),
            'input_type': nn.Embedding(num_type, num_model),
        })

        # output sequence
        self.query_coord_embedding = nn.Embedding(num_output_dof, num_model)
        self.query_pos_embedding = nn.Embedding(max_num_output, num_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            num_model, num_head, num_feedforward, dropout, activation, normalize_before, batch_first=True)
        encoder_norm = nn.LayerNorm(num_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

        # transformer decoder
        decoder_layers = TransformerDecoderLayer(
            num_model, num_head, num_feedforward, dropout, activation, normalize_before, batch_first=True)
        decoder_norm = nn.LayerNorm(num_model)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)

        # attach pointer
        self.vocab_head = nn.Linear(num_model, vocab_size)
        self.pointer_head = nn.Linear(num_model, num_model)   
        self.switch_head = nn.Linear(num_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _generate_pointer_mask(self, sz):
        r"""Generate a valid mask for the pointer. The masked positions are filled with 0.
            Unmasked positions are filled with 1.
        """
        switch = [3, 4, 5, 0, 1, 2]
        plank2plank_mask = torch.eye(6)[switch]
        plank2bbox_mask = torch.eye(6).repeat(self.max_num_output, 1)
        mask = plank2plank_mask.repeat(self.max_num_output, self.max_num_output)
        mask[:, :6] = plank2bbox_mask
        mask[:6, :] = 0
        return mask[:sz, :sz]

    def _embed_input(self, inputs):
        """Embeds flat input and adds position and coordinate information."""

        input_embeds = 0
        for key, value in inputs.items():
            if 'mask' in key:
                continue
            input_embeds += self.input_embeddings[key](value)

        return input_embeds

    def _embed_output(self, output):
        """Embeds flat input and adds position and coordinate information."""

        device = output.device
        batch_size, num_output_length = output.size(0), output.size(1)

        value_embeds = self.input_embeddings['input_value'](output)

        coords = torch.remainder(
            torch.arange(num_output_length, device=device), self.num_output_dof)
        coord_embeds = self.query_coord_embedding(coords)
        coord_embeds = coord_embeds.unsqueeze(0)

        positions = torch.div(
            torch.arange(num_output_length, device=device), self.num_output_dof, rounding_mode='floor')
        pos_embeds = self.query_pos_embedding(positions)
        pos_embeds = pos_embeds.unsqueeze(0)

        output_embeds = value_embeds + coord_embeds + pos_embeds

        # insert zero embedding at the beginning
        zero_embed = torch.zeros((batch_size, 1, self.num_model), device=device)
        output_embeds = torch.concat((zero_embed, output_embeds), dim=1)

        return output_embeds

    def _create_dist(self, outputs, eps=1e-6):
        """Outputs categorical dist for quantized vertex coordinates."""
        sz = outputs.size(1)
        
        # vocab dists (batch_size, seq_length, vocab_size)
        vocab_logits = self.vocab_head(outputs)

        # pointer dists (batch_size, seq_length, seq_length)
        pointer_feature = self.pointer_head(outputs)
        pointer_logits = torch.bmm(pointer_feature, outputs.transpose(1, 2))
        pointer_logits /= self.num_model
        
        #  attachment probability
        prob_logit = self.switch_head(outputs)
        prob = torch.sigmoid(prob_logit)

        if self.training:
    
            vocab_dists = F.log_softmax(vocab_logits, dim=-1)

            mask = (torch.triu(torch.ones(sz, sz, device=outputs.device)) == 1)
            pointer_logits.masked_fill_(mask.unsqueeze(0), eps)

            pointer_dists = F.log_softmax(pointer_logits, dim=-1)

            vocab_dists = vocab_dists + torch.log(torch.clamp(1 - prob, min=eps))
            pointer_dists = pointer_dists + torch.log(torch.clamp(prob, min=eps))

        else:

            vocab_dists = F.softmax(vocab_logits, dim=-1)

            if sz < 6:
                return vocab_dists

            mask = (torch.triu(torch.ones(sz, sz, device=outputs.device)) == 1)
            pointer_logits.masked_fill_(mask.unsqueeze(0), float('-inf'))

            pointer_dists = F.softmax(pointer_logits, dim=-1)

            vocab_dists = torch.mul(vocab_dists, 1 - prob)
            pointer_dists = torch.mul(pointer_dists, prob)

            pointer_mask = (self._generate_pointer_mask(sz) == 0).to(outputs.device)
            pointer_dists.masked_fill_(pointer_mask.unsqueeze(0), eps)

        dists = torch.cat((vocab_dists, pointer_dists), dim=-1)

        return dists

    def train_step(self, batch):
        """Pass batch through plank model and get log probabilities under model."""

        inputs = {key:value for key, value in batch.items() if key[:5]=='input'}

        input_mask = batch['input_mask']
        output_value = batch['output_value']
        output_label = batch['output_label']
        output_mask = batch['output_mask']

        # embed input
        input_embeds = self._embed_input(inputs)

        # embed output
        output_embeds = self._embed_output(output_value[:, :-1])

        memory = self.encoder(input_embeds, src_key_padding_mask=input_mask)

        # tgt mask
        tgt_mask = self._generate_square_subsequent_mask(output_embeds.size(1)).to(input_embeds.device)

        # pass through decoder
        hiddens = self.decoder(output_embeds, memory, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=output_mask,
                               memory_key_padding_mask=input_mask)

        # create categorical distribution
        dists = self._create_dist(hiddens)

        logits = dists.transpose(1, 2)

        loss = F.nll_loss(logits, output_label, ignore_index=self.token.PAD)

        predict = torch.argmax(logits, dim=1)

        valid = output_label != self.token.PAD
        correct = (valid * (predict == output_label)).sum()
        accuracy = float(correct) / (valid.sum() + 1e-10)

        rets = {}
        rets['loss'] = loss
        rets['accuracy'] = accuracy

        return rets

    def _sample(self, dists, samples):

        tokens = torch.argmax(dists[:, -1], -1, keepdim=True)

        pointers = torch.full_like(tokens, -1, dtype=torch.long)

        if torch.any(tokens >= self.vocab_size):

            batch_indices = torch.arange(len(tokens), device=tokens.device)

            tokens = tokens.flatten()
            pointers = pointers.flatten()

            mask = tokens >= self.vocab_size

            pointers[mask] = tokens[mask] - self.vocab_size
            tokens[mask] = samples[batch_indices[mask], tokens[mask] - self.vocab_size]

            tokens = tokens.unsqueeze(1)
            pointers = pointers.unsqueeze(1)

        return tokens, pointers

    def parse_sequence(self, sequence):
        valid_mask = torch.cumsum(sequence == self.token.END, 0) == 0
        valid_seq = sequence[valid_mask]

        num_plank = len(valid_seq) // self.num_output_dof
        bboxes = valid_seq[:num_plank * self.num_output_dof].reshape(-1, self.num_output_dof)

        return bboxes

    def eval_step(self, batch):
        """Autoregressive sampling."""

        inputs = {key:value for key, value in batch.items() if key[:5]=='input'}
        input_mask = inputs['input_mask']

        # embed inputs
        input_embeds = self._embed_input(inputs)

        batch_size = len(input_embeds)
        device = input_embeds.device

        memory = self.encoder(input_embeds, src_key_padding_mask=input_mask)

        output = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        attach = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        for _ in range(self.max_output_length):

            # embed decoder inputs
            output_embeds = self._embed_output(output)

            # tgt mask
            tgt_mask = self._generate_square_subsequent_mask(output.size(1)+1).to(device)

            # pass through decoder
            hiddens = self.decoder(output_embeds, memory, tgt_mask=tgt_mask,
                                   memory_key_padding_mask=input_mask)

            # create categorical distribution
            dists = self._create_dist(hiddens)

            # sample from the distribution
            next_output, next_attach = self._sample(dists, output)

            # update samples
            output = torch.concat((output, next_output), dim=1)
            attach = torch.concat((attach, next_attach), dim=1)

            if torch.all(torch.any(output == self.token.END, dim=1)):
                break

        predicts, groundtruths = [], []
        for i in range(batch_size):
            predict = self.parse_sequence(output[i])
            groundtruth = self.parse_sequence(batch['output_value'][i])

            predicts.append(predict)
            groundtruths.append(groundtruth)

        rets = {}
        rets['samples'] = output
        rets['attach'] = attach
        rets['predicts'] = predicts
        rets['groundtruths'] = groundtruths

        return rets

    def forward(self, batch):
        if self.training:
            outputs = self.train_step(batch)
        else:
            outputs = self.eval_step(batch)
        return outputs


def build_model(cfg):
    return PlankModel(
        cfg.MODEL.NUM_MODEL, cfg.MODEL.NUM_HEAD,
        cfg.MODEL.NUM_FEEDFORWARD, cfg.MODEL.DROPOUT,
        cfg.MODEL.ACTIVATION, cfg.MODEL.NORMALIZE_BEFORE,
        cfg.MODEL.NUM_ENCODER_LAYERS, cfg.MODEL.NUM_DECODER_LAYERS,
        cfg.DATA.NUM_VIEW, cfg.DATA.NUM_TYPE,
        cfg.DATA.NUM_INPUT_DOF, cfg.DATA.NUM_OUTPUT_DOF, 
        cfg.DATA.MAX_INPUT_LENGTH, cfg.DATA.MAX_OUTPUT_LENGTH,
        cfg.DATA.VOCAB_SIZE, cfg.TOKEN
    )

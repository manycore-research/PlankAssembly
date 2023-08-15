# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from plankassembly.position_encoding import PositionEmbeddingLearned


class PlankModel(nn.Module):

    def __init__(self,
                 num_model=512,
                 num_head=8,
                 num_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True,
                 num_decoder_layers=6,
                 vocab_size=514,
                 num_output_dof=6,
                 max_output_length=128,
                 token=None):
        super(PlankModel, self).__init__()

        max_num_output = math.ceil(max_output_length / num_output_dof)

        self.num_model = num_model
        self.max_output_length = max_output_length
        self.num_output_dof = num_output_dof
        self.max_num_output = max_num_output
        self.vocab_size = vocab_size

        self.token = token

        # input sequence
        self.pos_encoder = PositionEmbeddingLearned(num_model)

        # output sequence
        self.value_embedding = nn.Embedding(vocab_size, num_model)
        self.query_coord_embedding = nn.Embedding(num_output_dof, num_model)
        self.query_pos_embedding = nn.Embedding(max_num_output, num_model)

        # projection
        self.linear = nn.Linear(num_model, num_model)

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

    def _embed_input(self, features):
        # positional encoding
        pos_encoding = self.pos_encoder(features)
        features = features.permute(0, 2, 3, 4, 1)      # BxHxWxDxC
        pos_encoding = pos_encoding.permute(0, 2, 3, 4, 1)

        features = self.linear(features)
        features += pos_encoding

        # flatten features into sequence
        memory = features.flatten(1, 3)
        return memory

    def _embed_output(self, output):
        """Embeds flat input and adds position and coordinate information."""

        device = output.device
        batch_size, num_output_length = output.size(0), output.size(1)

        value_embeds = self.value_embedding(output)

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

    def train_step(self, features, targets):
        """Pass batch through plank model and get log probabilities under model."""
        device = features.device

        output = targets['output_value']
        mask = targets['output_mask']
        label = targets['output_label']

        memory = self._embed_input(features)

        # embed output
        output = self._embed_output(output[:, :-1])

        # tgt mask
        tgt_mask = self._generate_square_subsequent_mask(output.size(1)).to(device)

        # pass through decoder
        hiddens = self.decoder(output, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=mask)

        # create categorical distribution
        dists = self._create_dist(hiddens)

        logits = dists.transpose(1, 2)

        loss = F.nll_loss(logits, label, ignore_index=self.token.PAD)

        predict = torch.argmax(logits, dim=1)

        valid = label != self.token.PAD
        correct = (valid * (predict == label)).sum()
        accuracy = float(correct) / (valid.sum() + 1e-10)

        outputs = {}
        outputs['loss'] = loss
        outputs['accuracy'] = accuracy

        return outputs

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

    def _parse_sequence(self, sequence):
        valid_mask = torch.cumsum(sequence == self.token.END, 0) == 0
        valid_seq = sequence[valid_mask]

        num_plank = len(valid_seq) // self.num_output_dof
        bboxes = valid_seq[:num_plank * self.num_output_dof].reshape(-1, self.num_output_dof)

        return bboxes

    def eval_step(self, features, targets):
        """Autoregressive sampling."""
        batch_size = len(features)
        device = features.device

        memory = self._embed_input(features)

        samples = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        attach = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        for _ in range(self.max_output_length):

            # embed inputs
            decoder_inputs = self._embed_output(samples)

            # tgt mask
            tgt_mask = self._generate_square_subsequent_mask(samples.size(1)+1).to(device)

            # pass through decoder
            hiddens = self.decoder(decoder_inputs, memory, tgt_mask=tgt_mask)

            # create categorical distribution
            dists = self._create_dist(hiddens)

            # sample from the distribution
            sampled_tokens, pointers = self._sample(dists, samples)

            # update samples
            samples = torch.concat((samples, sampled_tokens), dim=1)
            attach = torch.concat((attach, pointers), dim=1)

            if torch.all(torch.any(samples == self.token.END, dim=1)):
                break

        predicts = []
        groundtruths = []
        for i in range(batch_size):
            predict = self._parse_sequence(samples[i])
            groundtruth = self._parse_sequence(targets['output_value'][i])

            predicts.append(predict)
            groundtruths.append(groundtruth)

        outputs = {}
        outputs['samples'] = samples
        outputs['predicts'] = predicts
        outputs['groundtruths'] = groundtruths

        return outputs

    def forward(self, feature, targets):
        if self.training:
            outputs = self.train_step(feature, targets)
        else:
            outputs = self.eval_step(feature, targets)
        return outputs


def build_model(cfg):
    return PlankModel(
        cfg.MODEL.HEAD.NUM_MODEL, cfg.MODEL.HEAD.NUM_HEAD,
        cfg.MODEL.HEAD.NUM_FEEDFORWARD, cfg.MODEL.HEAD.DROPOUT,
        cfg.MODEL.HEAD.ACTIVATION, cfg.MODEL.HEAD.NORMALIZE_BEFORE,
        cfg.MODEL.HEAD.NUM_DECODER_LAYERS, cfg.DATA.VOCAB_SIZE, 
        cfg.DATA.NUM_OUTPUT_DOF, cfg.DATA.MAX_OUTPUT_LENGTH, cfg.TOKEN
    )

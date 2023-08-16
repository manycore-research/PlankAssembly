# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (TransformerDecoder, TransformerDecoderLayer,
                      TransformerEncoder, TransformerEncoderLayer)

from polygen.pytorch_utils import min_value_of_dtype


class VertexModel(nn.Module):
    """Autoregressive generative model of quantized mesh vertices.

    Operates on flattened vertex sequences with a stopping token:

    [x_0, y_0, z_0, x_1, y_1, z_1, ..., x_n, y_n, z_n, STOP]

    Input vertex coordinates are embedded and tagged with learned coordinate and
    position indicators. A transformer encoder takes svgs as inputs. 
    A transformer decoder outputs logits for a quantized vertex distribution.
    """

    def __init__(self,
                 num_model=512,
                 num_head=8,
                 num_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 vocab_size=514,
                 num_view=3,
                 num_type=2,
                 num_line_dof=4,
                 num_vert_dof=3,
                 max_line_seq=1200,
                 max_vert_seq=460,
                 token=None,
                 **kwargs):
        super(VertexModel, self).__init__()

        max_num_input = math.ceil(max_line_seq / num_line_dof)
        max_num_output = math.ceil(max_vert_seq / num_vert_dof)

        self.num_model = num_model
        self.max_line_seq = max_line_seq
        self.max_vert_seq = max_vert_seq
        self.max_num_vertex = max_num_output
        self.vocab_size = vocab_size

        self.token = token

        # line embeddings
        self.line_embeddings = nn.ModuleDict({
            'line_value': nn.Embedding(vocab_size, num_model),
            'line_pos': nn.Embedding(max_num_input, num_model),
            'line_coord': nn.Embedding(num_line_dof, num_model),
            'line_view': nn.Embedding(num_view, num_model),
            'line_type': nn.Embedding(num_type, num_model),
        })

        # vertex embeddings
        self.query_coord_embedding = nn.Embedding(num_vert_dof, num_model)
        self.query_pos_embedding = nn.Embedding(max_num_output, num_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            num_model, num_head, num_feedforward, dropout, activation, normalize_before, batch_first=True)
        encoder_norm = nn.LayerNorm(num_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

        # Transformer decoder
        decoder_layers = TransformerDecoderLayer(
            num_model, num_head, num_feedforward, dropout, activation, normalize_before, batch_first=True)
        decoder_norm = nn.LayerNorm(num_model)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)

        self._project_to_logits = nn.Linear(num_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _embed_lines(self, lines):
        """Embeds flat input and adds position and coordinate information."""

        line_embeds = 0
        for key, value in lines.items():
            if 'mask' in key:
                continue
            line_embeds += self.line_embeddings[key](value)

        return line_embeds

    def _embed_vertices(self, vertices):
        """Embeds flat input and adds position and coordinate information."""

        device = vertices.device
        batch_size, num_vertex_length = vertices.size(0), vertices.size(1)

        value_embeds = self.line_embeddings['line_value'](vertices)

        coords = torch.remainder(
            torch.arange(num_vertex_length, device=device), 3)
        coord_embeds = self.query_coord_embedding(coords)
        coord_embeds = coord_embeds.unsqueeze(0)

        positions = torch.div(
            torch.arange(num_vertex_length, device=device), 3, rounding_mode='floor')
        pos_embeds = self.query_pos_embedding(positions)
        pos_embeds = pos_embeds.unsqueeze(0)

        vertex_embeds = value_embeds + coord_embeds + pos_embeds

        # insert zero embedding at the beginning
        zero_embed = torch.zeros((batch_size, 1, self.num_model), device=device)
        vertex_embeds = torch.concat((zero_embed, vertex_embeds), dim=1)

        return vertex_embeds

    def train_step(self, batch):
        """Pass batch through plank model and get log probabilities under model."""

        lines = {key:value for key, value in batch.items() if key.startswith('line')}

        line_mask = batch['line_mask']
        vertex = batch['vertex']
        vertex_mask = batch['vertex_mask']

        device = line_mask.device

        # embed lines
        line_embeddings = self._embed_lines(lines)

        # embed vertices
        vertex_embeddings = self._embed_vertices(vertex[:, :-1])

        memory = self.encoder(line_embeddings, src_key_padding_mask=line_mask)

        # tgt mask
        tgt_mask = self.generate_square_subsequent_mask(vertex_embeddings.size(1)).to(device)

        # pass through decoder
        hiddens = self.decoder(vertex_embeddings, memory, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=vertex_mask,
                               memory_key_padding_mask=line_mask)

        # create categorical distribution
        logits = self._project_to_logits(hiddens)

        logits = logits.transpose(1, 2)

        loss = F.cross_entropy(logits, vertex, ignore_index=self.token['PAD'])

        predict = torch.argmax(logits, dim=1)

        valid = vertex != self.token['PAD']
        correct = (valid * (predict == vertex)).sum()
        accuracy = float(correct) / (valid.sum() + 1e-10)

        outputs = {}
        outputs['loss'] = loss
        outputs['accuracy'] = accuracy

        return outputs

    def eval_step(self, batch):
        """Autoregressive sampling."""

        lines = {key:value for key, value in batch.items() if key.startswith('line')}
        line_mask = batch['line_mask']

        batch_size = len(line_mask)
        device = line_mask.device

        # embed inputs
        line_embeddings = self._embed_lines(lines)

        memory = self.encoder(line_embeddings, src_key_padding_mask=line_mask)

        samples = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        for _ in range(self.max_vert_seq):

            # embed inputs
            decoder_inputs = self._embed_vertices(samples)

            # tgt mask
            tgt_mask = self.generate_square_subsequent_mask(samples.size(1)+1).to(device)

            # pass through decoder
            hiddens = self.decoder(decoder_inputs, memory, tgt_mask=tgt_mask,
                                   memory_key_padding_mask=line_mask)

            # create categorical distribution
            dists = self._project_to_logits(hiddens)

            tokens = torch.argmax(dists[:, -1], -1, keepdim=True)

            # update samples
            samples = torch.concat((samples, tokens), dim=1)

            if torch.all(torch.any(samples == self.token['END'], dim=1)):
                break

        return samples

    def forward(self, data):
        if self.training:
            outputs = self.train_step(data)
        else:
            outputs = self.eval_step(data)
        return outputs


class FaceModel(nn.Module):
    """Autoregressive generative model of n-gon meshes.

    Operates on sets of input vertices as well as flattened face sequences with
    new face and stopping tokens:

    [f_0^0, f_0^1, f_0^2, NEW, f_1^0, f_1^1, ..., STOP]

    Input vertices are encoded using a Transformer encoder.

    Input face sequences are embedded and tagged with learned position indicators,
    as well as their corresponding vertex embeddings. A transformer decoder
    outputs a pointer which is compared to each vertex embedding to obtain a
    distribution over vertex indices.
    """

    def __init__(self,
                 num_model=512,
                 num_head=8,
                 num_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_vert_dof=3,
                 max_num_vert=151,
                 max_face_seq=580,
                 vocab_size=512,
                 token=None,
                 **kwargs):
        super(FaceModel, self).__init__()

        self.num_model = num_model
        self.num_vert_dof = num_vert_dof
        self.max_num_vert = max_num_vert
        self.max_face_seq = max_face_seq
        self.vocab_size = vocab_size

        self.token = token

        # vertex embeddings
        self.value_embedding = nn.ModuleList(
            [nn.Embedding(vocab_size, num_model) for _ in range(self.num_vert_dof)])
        self.pos_embedding = nn.Embedding(max_num_vert, num_model)
        self.token_embedding = nn.Embedding(len(token), num_model)

        # face embeddings
        self.query_pos_embedding = nn.Embedding(max_face_seq, num_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            num_model, num_head, num_feedforward, dropout, activation, normalize_before, batch_first=True)
        encoder_norm = nn.LayerNorm(num_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

        # Transformer decoder
        decoder_layers = TransformerDecoderLayer(
            num_model, num_head, num_feedforward, dropout, activation, normalize_before, batch_first=True)
        decoder_norm = nn.LayerNorm(num_model)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)

        # attach pointer
        self._project_to_pointers = nn.Linear(num_model, num_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _embed_vertices(self, vertices, vertices_mask):
        """Embeds flat input and adds position and coordinate information."""

        device = vertices.device

        batch_size = len(vertices)

        value_embeds = 0
        for i in range(self.num_vert_dof):
            value_embeds += self.value_embedding[i](vertices[..., i])

        positions = torch.arange(self.max_num_vert, device=device)
        pos_embeds = self.pos_embedding(positions)
        pos_embeds = pos_embeds.unsqueeze(0)

        vertex_embeds = value_embeds + pos_embeds

        token_embeds = self.token_embedding.weight.unsqueeze(0)
        token_embeds = token_embeds.expand(batch_size, -1, -1)

        vertex_embeddings = torch.cat((token_embeds, vertex_embeds), dim=1)

        vertex_embeddings = self.encoder(vertex_embeddings, src_key_padding_mask=vertices_mask)

        return vertex_embeddings

    def _embed_faces(self, vertex_embeddings, faces):
        """Embeds flat input and adds position and coordinate information."""

        device = faces.device
        batch_size, num_output_length = faces.size(0), faces.size(1)

        faces = faces.unsqueeze(-1).repeat(1, 1, self.num_model)

        face_embeddings = torch.gather(vertex_embeddings, 1, faces)

        positions = torch.arange(num_output_length, device=device)
        pos_embeds = self.query_pos_embedding(positions)
        pos_embeds = pos_embeds.unsqueeze(0)

        output_embeds = face_embeddings + pos_embeds

        # insert zero embedding at the beginning
        zero_embed = torch.zeros((batch_size, 1, self.num_model), device=device)
        output_embeds = torch.concat((zero_embed, output_embeds), dim=1)

        return output_embeds

    def train_step(self, inputs):
        """Pass batch through plank model and get log probabilities under model."""

        vertex = inputs['vertex']
        vertex_mask = inputs['vertex_mask']

        faces = inputs['face']
        faces_mask = inputs['face_mask']

        # embed vertices
        vertex_embeddings = self._embed_vertices(vertex, vertex_mask)

        # embed faces
        face_embeddings = self._embed_faces(vertex_embeddings, faces[:, :-1])

        # tgt mask
        tgt_mask = self.generate_square_subsequent_mask(face_embeddings.size(1)).to(vertex.device)

        # pass through decoder
        hiddens = self.decoder(face_embeddings, vertex_embeddings,
                               tgt_mask=tgt_mask,
                               tgt_key_padding_mask=faces_mask,
                               memory_key_padding_mask=vertex_mask)

        # create categorical distribution
        pointers = self._project_to_pointers(hiddens)

        logits = torch.bmm(vertex_embeddings, pointers.transpose(1, 2))

        # label: N x T
        loss = F.cross_entropy(logits, faces, ignore_index=self.token['PAD'])

        predicts = torch.argmax(logits, dim=1)

        valid = faces != self.token['PAD']
        correct = (valid * (predicts == faces)).sum()
        accuracy = float(correct) / (valid.sum() + 1e-10)

        outputs = {}
        outputs['loss'] = loss
        outputs['accuracy'] = accuracy

        return outputs

    def select_next(self, embedding, pointer, input_mask):
        pointer = pointer.transpose(1, 2)
        logits = torch.bmm(embedding, pointer[..., -1:])
        logits = logits.masked_fill(input_mask.unsqueeze(-1), min_value_of_dtype(logits.dtype))
        next_token = torch.argmax(logits, dim=1)
        return next_token

    def eval_step(self, inputs):
        """Autoregressive sampling."""

        vertex = inputs['vertex']
        vertex_mask = inputs['vertex_mask']

        batch_size = len(vertex)
        device = vertex.device

        # embed vertices
        vertex_embeddings = self._embed_vertices(vertex, vertex_mask)

        samples = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        for _ in range(self.max_face_seq):

            # embed faces
            face_embeddings = self._embed_faces(vertex_embeddings, samples)

            # tgt mask
            tgt_mask = self.generate_square_subsequent_mask(samples.size(1)+1).to(device)

            # pass through decoder
            hiddens = self.decoder(face_embeddings, vertex_embeddings,
                                   tgt_mask=tgt_mask,
                                   memory_key_padding_mask=vertex_mask)

            # create categorical distribution
            pointers = self._project_to_pointers(hiddens)

            tokens = self.select_next(vertex_embeddings, pointers, vertex_mask)

            samples = torch.concat((samples, tokens), dim=1)

            if torch.all(torch.any(samples == self.token['END'], dim=1)):
                break

        return samples

    def forward(self, data):
        if self.training:
            outputs = self.train_step(data)
        else:
            outputs = self.eval_step(data)
        return outputs

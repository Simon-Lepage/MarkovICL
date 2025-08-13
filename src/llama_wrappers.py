from dataclasses import dataclass
import torch

from transformers import LlamaModel


@dataclass
class TFOutput:
    logits: torch.Tensor
    attentions: torch.Tensor = None


class DefaultLlama(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = LlamaModel(config)

    def forward(self, input_ids, output_attentions=False):
        emb = self.model.embed_tokens(input_ids)
        out = self.model(inputs_embeds=emb, output_attentions=output_attentions)
        logits = torch.einsum(
            "bld,vd->blv", out.last_hidden_state, self.model.embed_tokens.weight
        )

        return TFOutput(logits=logits, attentions=out.attentions)


class OrthogonalLlama(torch.nn.Module):
    def __init__(self, config):
        assert config.vocab_size == 0
        super().__init__()

        self.model = LlamaModel(config)

    @staticmethod
    def _ortho_vocab(B, V, D, device):
        _d = D // 2
        # Batched random orthogonal embeddings
        emb_dict = torch.randn(B, max(V, _d), _d, dtype=torch.float32, device=device)
        emb_dict, _ = torch.linalg.qr(emb_dict)
        emb_dict = emb_dict[:, :V, :_d]  # V vectors of size D
        # Now pad with zeros : B x V x D -> B x V x 2D
        emb_dict = torch.cat([emb_dict, torch.zeros(B, V, _d, device=device)], dim=-1)
        return emb_dict

    def forward(self, input_ids, output_attentions=False):
        voc = self._ortho_vocab(
            B=input_ids.shape[0],
            V=input_ids.max() + 1,  # If 0-9, we want to have 10
            D=self.model.config.hidden_size,
            device=input_ids.device,
        )

        row_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        row_index = row_index.view(-1, 1)

        emb = voc[row_index, input_ids]

        out = self.model(inputs_embeds=emb, output_attentions=output_attentions)

        logits = torch.einsum("bld,bvd->blv", out.last_hidden_state, voc)

        return TFOutput(logits=logits, attentions=out.attentions)

import math
import os

import pytorch_lightning as pl
import numpy as np
import matplotlib as plt
import random

import torch
import wandb

from scripts.models import DeepROTransformer


class AttentionMatrixCallback(pl.Callback):
    def __init__(self, test_batches, layer=0, total_samples=32):
        self.test_data = test_batches
        self.total_samples = total_samples
        self.layer = layer

    def _get_attns(self, model):
        idxs = [random.sample(range(self.test_data[0].shape[0]), k=self.total_samples)]
        x_c, y_c, x_t, y_t = [i[idxs].detach().to(model.device) for i in self.test_data]
        enc_attns, dec_attns = None, None
        # save memory by doing inference 1 example at a time
        for i in range(self.total_samples):
            x_ci = x_c[i].unsqueeze(0)
            y_ci = y_c[i].unsqueeze(0)
            x_ti = x_t[i].unsqueeze(0)
            y_ti = y_t[i].unsqueeze(0)
            with torch.no_grad():
                *_, (enc_self_attn, dec_cross_attn) = model(
                    x_ci, y_ci, x_ti, y_ti, output_attn=True
                )
            if enc_attns is None:
                enc_attns = [[a] for a in enc_self_attn]
            else:
                for cum_attn, attn in zip(enc_attns, enc_self_attn):
                    cum_attn.append(attn)
            if dec_attns is None:
                dec_attns = [[a] for a in dec_cross_attn]
            else:
                for cum_attn, attn in zip(dec_attns, dec_cross_attn):
                    cum_attn.append(attn)

        # re-concat over batch dim, avg over batch dim
        if enc_attns:
            enc_attns = [torch.cat(a, dim=0) for a in enc_attns][self.layer].mean(0)
        else:
            enc_attns = None
        if dec_attns:
            dec_attns = [torch.cat(a, dim=0) for a in dec_attns][self.layer].mean(0)
        else:
            dec_attns = None
        return enc_attns, dec_attns

    def _make_imgs(self, attns, img_title_prefix):
        heads = [i for i in range(attns.shape[0])] + ["avg", "sum"]
        imgs = []
        for head in heads:
            if head == "avg":
                a_head = attns.mean(0)
            elif head == "sum":
                a_head = attns.sum(0)
            else:
                a_head = attns[head]

            a_head /= torch.max(a_head, dim=-1)[0].unsqueeze(1)

            imgs.append(
                wandb.Image(
                    show_image(
                        a_head.cpu().numpy(),
                        f"{img_title_prefix} Head {str(head)}",
                        tick_spacing=a_head.shape[-2],
                        cmap="Blues",
                    )
                )
            )
        return imgs

    def _pos_sim_scores(self, embedding, seq_len, device):
        if embedding.position_emb == "t2v":
            inp = torch.arange(seq_len).float().to(device).view(1, -1, 1)
            encoder_embs = embedding.local_emb(inp)[0, :, 1:]
        elif embedding.position_emb == "abs":
            encoder_embs = embedding.local_emb(torch.arange(seq_len).to(device).long())
        cos_sim = torch.nn.CosineSimilarity(dim=0)
        scores = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(0, i + 1):
                sim = cos_sim(encoder_embs[i], encoder_embs[j])
                scores[i, j] = sim
                scores[j, i] = sim
        return scores

    def on_validation_end(self, trainer, model):
        self_attns, cross_attns = self._get_attns(model)

        if self_attns is not None:
            self_attn_imgs = self._make_imgs(
                self_attns, f"Self Attn, Layer {self.layer},"
            )
            trainer.logger.experiment.log(
                {"test/self_attn": self_attn_imgs, "global_step": trainer.global_step}
            )
        if cross_attns is not None:
            cross_attn_imgs = self._make_imgs(
                cross_attns, f"Cross Attn, Layer {self.layer},"
            )
            trainer.logger.experiment.log(
                {"test/cross_attn": cross_attn_imgs, "global_step": trainer.global_step}
            )

        enc_emb_sim = self._pos_sim_scores(
            model.spacetimeformer.enc_embedding,
            seq_len=self.test_data[1].shape[1],
            device=model.device,
        )
        dec_emb_sim = self._pos_sim_scores(
            model.spacetimeformer.dec_embedding,
            seq_len=self.test_data[3].shape[1],
            device=model.device,
        )
        emb_sim_imgs = [
            wandb.Image(
                show_image(
                    enc_emb_sim,
                    f"Encoder Position Emb. Similarity",
                    tick_spacing=enc_emb_sim.shape[-1],
                    cmap="Greens",
                )
            ),
            wandb.Image(
                show_image(
                    dec_emb_sim,
                    f"Decoder Position Emb. Similarity",
                    tick_spacing=dec_emb_sim.shape[-1],
                    cmap="Greens",
                )
            ),
        ]
        trainer.logger.experiment.log(
            {"test/pos_embs": emb_sim_imgs, "global_step": trainer.global_step}
        )

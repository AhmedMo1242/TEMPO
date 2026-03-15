import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from modules import (
    BiLSTMLayer,
    TemporalConv,
    TAPEAdapter,
    MSTCN_Skeleton,
    HybridBackend,
)
from modules.stochastic_depth import StochasticDepth
from modules.cross_stream_consistency import CrossStreamConsistencyLoss
from modules.visual_extractor import CoSign2s


class KLdis(nn.Module):
    def __init__(self, T=1):
        super().__init__()
        self.kdloss = nn.KLDivLoss(reduction="batchmean")
        self.T = T

    def forward(self, view1_logits, view2_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        v1 = torch.clamp(view1_logits[:, :, start_idx:] / self.T, min=-50, max=50)
        v2 = torch.clamp(view2_logits[:, :, start_idx:] / self.T, min=-50, max=50)

        log_p1 = F.log_softmax(v1, dim=-1).view(-1, v1.shape[-1])
        p2 = F.softmax(v2, dim=-1).view(-1, v2.shape[-1])

        loss = self.kdloss(log_p1, p2) * (self.T**2)
        return loss


class NormBothLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormBothLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        x = torch.nan_to_num(x, 0.0)
        outputs = torch.matmul(
            F.normalize(x, dim=-1, eps=1e-5), F.normalize(self.weight, dim=0, eps=1e-5)
        )
        return outputs


class TwoStream_Cosign(nn.Module):
    def __init__(
        self,
        visual_args,
        gloss_dict,
        conv_type,
        loss_weights,
        norm_scale=24,
        num_layers=1,
        use_tape=False,
        use_mstcn=False,
        use_hybrid_backend=False,
        adapter_args={"adapter_channels": 64},
        hybrid_args={"num_transformer_layers": 2, "num_heads": 8, "dropout": 0.1},
        decoder_args={
            "beam_width": 20,
            "length_penalty": 0.15,
            "temperature": 1.0,
            "use_tta": False,
        },
        use_ssd=False,
        use_csc=False,
        **kwargs,  # Accept (and ignore) legacy config keys
    ) -> None:
        super().__init__()

        self.apply_CR = True if "CR_args" in visual_args else False
        self.visual_module = CoSign2s(**visual_args)
        hidden_size = self.visual_module.out_size
        self.num_classes = len(gloss_dict["id2gloss"]) + 1
        self.decoder = utils.Decode(
            gloss_dict, self.num_classes, "beam", **decoder_args
        )
        self.use_tape = use_tape
        self.use_mstcn = use_mstcn
        self.use_hybrid_backend = use_hybrid_backend
        self.use_ssd = use_ssd
        self.use_csc = use_csc

        self.stream_configs = {
            "static": {"input_dim": 256 * 4},
            "motion": {"input_dim": 256 * 4},
            "fusion": {"input_dim": hidden_size},
        }
        for name, config in self.stream_configs.items():
            # TAPE Adapter
            if self.use_tape:
                tape = TAPEAdapter(config["input_dim"], **adapter_args)
                setattr(self, f"tape_{name}", tape)

            # Temporal Module: MSTCN or TemporalConv
            if self.use_mstcn:
                conv1d = MSTCN_Skeleton(config["input_dim"], hidden_size)
            else:
                conv1d = TemporalConv(config["input_dim"], hidden_size, conv_type)

            # Contextual Module: Hybrid BiLSTM+Transformer or plain BiLSTM
            if self.use_hybrid_backend:
                contextual_module = HybridBackend(
                    rnn_type="LSTM",
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=True,
                    **hybrid_args,
                )
            else:
                contextual_module = BiLSTMLayer(
                    rnn_type="LSTM",
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=True,
                )

            classifier = NormBothLinear(hidden_size, self.num_classes)
            setattr(self, f"conv1d_{name}", conv1d)
            setattr(self, f"contextual_module_{name}", contextual_module)
            setattr(self, f"classifier_{name}", classifier)

        # CSC: Cross-Stream Consistency
        if self.use_csc:
            self.csc_module = CrossStreamConsistencyLoss(temperature=0.1)
            self.csc_loss_weight = 0.05

        # SSD: Stochastic Sequence Depth
        if self.use_ssd:
            self.ssd_module = StochasticDepth(prob=0.2)

        self.loss = {
            "ctc": nn.CTCLoss(reduction="none", zero_infinity=True),
            "kl": KLdis(),
        }
        self.loss_weights = loss_weights
        self.norm_scale = norm_scale

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def forward_contextual(
        self,
        framewise,
        len_x,
        conv1d_module,
        contextual_module,
        classifier,
        tape_module=None,
        ssd_module=None,
    ):
        # TAPE Adapter
        if tape_module is not None:
            framewise = tape_module(framewise)

        # Temporal Convolution (MSTCN or TemporalConv)
        with torch.cuda.amp.autocast(enabled=False):
            conv1d_input = framewise.transpose(1, 2).float()
            if ssd_module is not None:
                conv1d_ret = ssd_module(conv1d_input, lambda x: conv1d_module(x, len_x))
            else:
                conv1d_ret = conv1d_module(conv1d_input, len_x)

        conv1d_feat = conv1d_ret["visual_feat"].transpose(0, 1)  # B, T, C
        feat_len = conv1d_ret["feat_len"]

        # Clamp before contextual module to prevent NaNs/Inf
        if not torch.isfinite(conv1d_feat).all():
            conv1d_feat = torch.nan_to_num(
                conv1d_feat, nan=0.0, posinf=1e4, neginf=-1e4
            )

        # Contextual Module (Hybrid BiLSTM+Transformer or BiLSTM)
        with torch.cuda.amp.autocast(enabled=False):
            contextual_input = conv1d_feat.transpose(0, 1).float()
            contextual_feat = contextual_module(contextual_input, feat_len)[
                "predictions"
            ]

        if not torch.isfinite(contextual_feat).all():
            contextual_feat = torch.nan_to_num(
                contextual_feat, nan=0.0, posinf=1e4, neginf=-1e4
            )
        contextual_feat = contextual_feat.transpose(0, 1)  # B, T, C

        # Layer norm before classifier for numerical stability
        contextual_feat = F.layer_norm(contextual_feat, contextual_feat.shape[-1:])
        conv1d_feat = F.layer_norm(conv1d_feat, conv1d_feat.shape[-1:])

        # Classification: conv head (Y_c) and seq head (Y_s)
        conv1d_logits = classifier(conv1d_feat.transpose(0, 1))
        seq_logits = classifier(contextual_feat.transpose(0, 1))

        return {
            "conv_logits": conv1d_logits.transpose(0, 1),  # B, T, C
            "seq_logits": seq_logits.transpose(0, 1),  # B, T, C
            "feat_len": feat_len.to(conv1d_logits.device),
            "seq_feat": contextual_feat,  # (B, T, C) for CSC
        }

    def forward(self, inputs_dict):
        x, len_x = inputs_dict["x"], inputs_dict["len_x"]
        visual_ret = self.visual_module(x, len_x)

        if self.apply_CR and self.training:
            results = {}
            active_streams = [
                s
                for s in self.stream_configs.keys()
                if any(
                    k.endswith(f"_{s}") and self.loss_weights.get(k, 0) > 0
                    for k in self.loss_weights.keys()
                )
            ]
            if not active_streams:
                active_streams = ["fusion"]

            for stream_type in active_streams:
                view1, view2 = (
                    visual_ret[f"view1_{stream_type}"],
                    visual_ret[f"view2_{stream_type}"],
                )
                conv1d_module = getattr(self, f"conv1d_{stream_type}")
                contextual_module = getattr(self, f"contextual_module_{stream_type}")
                classifier = getattr(self, f"classifier_{stream_type}")
                tape_module = (
                    getattr(self, f"tape_{stream_type}") if self.use_tape else None
                )
                ssd_module = self.ssd_module if self.use_ssd else None

                for view_name, view_feat in [("view1", view1), ("view2", view2)]:
                    ret = self.forward_contextual(
                        view_feat,
                        len_x,
                        conv1d_module,
                        contextual_module,
                        classifier,
                        tape_module,
                        ssd_module=ssd_module,
                    )
                    results[f"{view_name}_{stream_type}_conv_logits"] = ret[
                        "conv_logits"
                    ]
                    results[f"{view_name}_{stream_type}_seq_logits"] = ret["seq_logits"]
                    results[f"{view_name}_{stream_type}_feat_len"] = ret["feat_len"]
                    results[f"{view_name}_{stream_type}_seq_feat"] = ret["seq_feat"]

            res_key = (
                "view1_fusion_feat_len"
                if "view1_fusion_feat_len" in results
                else f"view1_{active_streams[0]}_feat_len"
            )
            results["feat_len"] = results[res_key]
            return results
        else:
            fusion = visual_ret["fusion"]
            ret = self.forward_contextual(
                fusion,
                len_x,
                self.conv1d_fusion,
                self.contextual_module_fusion,
                self.classifier_fusion,
                self.tape_fusion if self.use_tape else None,
            )
            return {
                "conv_logits": ret["conv_logits"],
                "seq_logits": ret["seq_logits"],
                "feat_len": ret["feat_len"],
            }

    def decode(self, ret_dict):
        feat_len = ret_dict["feat_len"]
        if isinstance(feat_len, torch.Tensor):
            feat_len = feat_len.cpu().int()
        conv_sents = self.decoder.decode(
            ret_dict["conv_logits"] * self.norm_scale,
            feat_len,
            batch_first=True,
            probs=False,
        )
        seq_sents = self.decoder.decode(
            ret_dict["seq_logits"] * self.norm_scale,
            feat_len,
            batch_first=True,
            probs=False,
        )
        return {
            "conv_sents_fusion": conv_sents,
            "recognized_sents_fusion": seq_sents,
        }

    def get_ctc_loss(self, no_scale_logits, label, feat_len, label_len):
        log_probs = (no_scale_logits * self.norm_scale).log_softmax(-1)
        log_probs = log_probs.transpose(0, 1)  # [B, T, C] -> [T, B, C]
        return self.loss["ctc"](
            log_probs,
            label.cpu(),
            feat_len.cpu(),
            label_len.cpu(),
        ).mean()

    def get_loss(self, ret_dict, inputs_dict):
        _dev = next(
            (v.device for v in ret_dict.values() if isinstance(v, torch.Tensor)),
            torch.device("cpu"),
        )
        loss = torch.zeros(1, device=_dev).squeeze()
        loss_dict = {}

        label = inputs_dict.get("label", None)
        label_lgt = inputs_dict.get("label_lgt", None)

        if label is None or (
            isinstance(label, (list, torch.Tensor)) and len(label) == 0
        ):
            return loss, loss_dict
        if not isinstance(label, torch.Tensor):
            return loss, loss_dict

        # Inference-mode fallback (model.training == False)
        if "conv_logits" in ret_dict and "view1_fusion_conv_logits" not in ret_dict:
            feat_len = ret_dict.get("feat_len")
            if feat_len is not None and label_lgt is not None:
                w = self.loss_weights
                for logits_key, loss_key in [
                    ("conv_logits", "CR_ConvCTC_fusion"),
                    ("seq_logits", "CR_SeqCTC_fusion"),
                ]:
                    weight = w.get(loss_key, 0)
                    if weight > 0 and logits_key in ret_dict:
                        ctc = self.get_ctc_loss(
                            ret_dict[logits_key], label, feat_len, label_lgt
                        )
                        loss = loss + weight * ctc
                        loss_dict[loss_key] = ctc
            return loss, loss_dict

        # CSC Cross-Stream Consistency Loss
        if self.use_csc and self.training:
            feat_static = ret_dict.get("view1_static_seq_feat")
            feat_motion = ret_dict.get("view1_motion_seq_feat")
            if feat_static is not None and feat_motion is not None:
                csc_loss = self.csc_module(feat_static, feat_motion)
                loss = loss + self.csc_loss_weight * csc_loss
                loss_dict["csc"] = csc_loss

        for k, weight in self.loss_weights.items():
            if weight == 0:
                continue
            temp_loss = 0

            parts = k.split("_")
            if len(parts) < 3:
                continue
            loss_type = parts[1]
            stream_type = parts[2]

            def has_keys(*names):
                return all(n in ret_dict for n in names)

            if loss_type in ["ConvCTC", "SeqCTC"]:
                logits_key = "conv_logits" if loss_type == "ConvCTC" else "seq_logits"
                k1 = f"view1_{stream_type}_{logits_key}"
                k1len = f"view1_{stream_type}_feat_len"
                k2 = f"view2_{stream_type}_{logits_key}"
                k2len = f"view2_{stream_type}_feat_len"
                if not has_keys(k1, k1len, k2, k2len):
                    continue
                view1_loss = self.get_ctc_loss(
                    ret_dict[k1], label, ret_dict[k1len], label_lgt
                )
                view2_loss = self.get_ctc_loss(
                    ret_dict[k2], label, ret_dict[k2len], label_lgt
                )
                temp_loss = (view1_loss + view2_loss) * 0.5 * weight
            elif loss_type in ["Conv", "Seq"]:
                logits_key = "conv_logits" if loss_type == "Conv" else "seq_logits"
                k1 = f"view1_{stream_type}_{logits_key}"
                k2 = f"view2_{stream_type}_{logits_key}"
                if not has_keys(k1, k2):
                    continue
                view1_logits = ret_dict[k1] * self.norm_scale
                view2_logits = ret_dict[k2] * self.norm_scale
                kl_loss1 = self.loss["kl"](view1_logits, view2_logits)
                kl_loss2 = self.loss["kl"](view2_logits, view1_logits)
                temp_loss = (kl_loss1 + kl_loss2) * 0.5 * weight

            loss += temp_loss
            loss_dict[k] = temp_loss

        return loss, loss_dict

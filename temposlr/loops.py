import os
import csv
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from temposlr.evaluation.wer import compute_wer, load_csv_as_dict


def seq_train(
    loader,
    model,
    optimizer,
    device,
    epoch_idx,
    recoder,
    clip_grad=5.0,
    max_batches=None,
    scaler=None,
    **kwargs,
):
    model.train()
    loss_value = []
    clr = [group["lr"] for group in optimizer.optimizer.param_groups]
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler()

    for batch_idx, data in enumerate(tqdm(loader)):
        if max_batches is not None and batch_idx >= max_batches:
            recoder.print_log(
                f"Reached max_batches ({max_batches}), stopping training for this epoch."
            )
            break

        data = device.dict_data_to_device(data)

        with torch.cuda.amp.autocast():  # Mixed Precision
            ret_dict = model(data)
            if hasattr(model, "module"):
                loss, loss_details = model.module.get_loss(ret_dict, data)
            else:
                loss, loss_details = model.get_loss(ret_dict, data)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            recoder.print_log(f"Loss is NaN/Inf at batch {batch_idx}, skipping.")
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Unscale for gradient clipping & logging
        scaler.unscale_(optimizer.optimizer)

        # Sanitize gradients before clipping (prevents NaN propagation)
        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
                )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_grad
        )

        if batch_idx % recoder.log_interval == 0:
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                recoder.print_log(
                    f"WARNING: Grad norm is {grad_norm} at batch {batch_idx}"
                )

        scaler.step(optimizer.optimizer)
        scaler.update()

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                f"\tEpoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. Loss: {loss.item():.2f}  GradNorm: {grad_norm:.4f}  lr:{clr[0]:.6f}"
            )
            recoder.print_log(
                "\t"
                + ", ".join(
                    [
                        f"{k}: {v.item() if hasattr(v, 'item') else v:.2f}"
                        for k, v in loss_details.items()
                    ]
                )
            )

    optimizer.scheduler.step()
    avg_loss = np.mean(loss_value)
    recoder.print_log(
        f"\t[Epoch {epoch_idx} DONE] Mean train loss: {avg_loss:.4f}  "
        f"LR: {clr[0]:.6f}"
    )
    return loss_value


def seq_eval(
    cfg,
    loader,
    model,
    device,
    mode,
    epoch,
    work_dir,
    recoder,
    task,
    evaluate_tool="python",
    max_batches=None,
):
    model.eval()
    total_info = []
    total_sent_conv = []  # after MSTCN
    total_sent_seq = []  # after BiLSTM+Transformer
    val_loss_values = []

    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        if max_batches is not None and batch_idx >= max_batches:
            recoder.print_log(
                f"Reached eval max_batches ({max_batches}), stopping evaluation early."
            )
            break
        data = device.dict_data_to_device(data)
        with torch.no_grad():
            if hasattr(model, "module"):
                ret_dict = model.module(data)
                decoded_dict = model.module.decode(ret_dict)
                val_loss, _ = model.module.get_loss(ret_dict, data)
            else:
                ret_dict = model(data)
                decoded_dict = model.decode(ret_dict)
                val_loss, _ = model.get_loss(ret_dict, data)

        if isinstance(val_loss, torch.Tensor):
            if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                val_loss_values.append(val_loss.item())
        elif isinstance(val_loss, (int, float)) and not (
            np.isnan(val_loss) or np.isinf(val_loss)
        ):
            val_loss_values.append(float(val_loss))

        total_info += [fn.split("|")[0] for fn in data["origin_info"]]
        total_sent_conv += decoded_dict["conv_sents_fusion"]
        total_sent_seq += decoded_dict["recognized_sents_fusion"]

    avg_val_loss = np.mean(val_loss_values) if val_loss_values else float("nan")
    recoder.print_log(
        f"Epoch {epoch} [{mode}] Val Loss: {avg_val_loss:.4f}",
        f"{work_dir}/{mode}.txt",
    )

    # ── Build prediction dicts from decoded sentences ─────────────────────────
    def _build_pred_dict(id_list, sent_list):
        pred = {}
        for vid_id, sent in zip(id_list, sent_list):
            words = (
                [w[0] for w in sent]
                if sent and isinstance(sent[0], (list, tuple))
                else list(sent)
            )
            pred[vid_id] = " ".join(words)
        return pred

    pred_conv = _build_pred_dict(total_info, total_sent_conv)
    pred_seq = _build_pred_dict(total_info, total_sent_seq)

    # ── Always write prediction CSVs ─────────────────────────────────────────
    def _write_csv(pred_dict, tag):
        try:
            sorted_dict = dict(sorted(pred_dict.items(), key=lambda x: int(x[0])))
        except (ValueError, TypeError):
            sorted_dict = dict(sorted(pred_dict.items()))
        csv_path = os.path.join(work_dir, f"epoch{epoch}_{mode}_{tag}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "gloss"])
            for vid_id, gloss in sorted_dict.items():
                w.writerow([vid_id, gloss])
        return csv_path

    csv_conv = _write_csv(pred_conv, "conv")
    csv_seq = _write_csv(pred_seq, "seq")
    recoder.print_log(f"[{mode}] Saved CSVs: {csv_conv}, {csv_seq}")

    # ── test mode: no reference → just return CSV path ────────────────────────
    if mode == "test":
        return csv_seq

    # ── dev/train mode: compute WER against reference CSV ─────────────────────
    ref_csv = cfg.dataset_info.get("ref_csv", None)
    if ref_csv is None:
        recoder.print_log(f"[WARN] No ref_csv in dataset config, WER=100.0")
        return 100.0

    ref_dict = load_csv_as_dict(ref_csv)

    wer_conv, det_conv = compute_wer(ref_dict, pred_conv)
    wer_seq, det_seq = compute_wer(ref_dict, pred_seq)

    best_wer = min(wer_conv, wer_seq)
    best_label = "MSTCN" if wer_conv <= wer_seq else "SEQ"

    recoder.print_log(
        f"Epoch {epoch} [{mode}] | "
        f"ValLoss: {avg_val_loss:.4f}  "
        f"MSTCN: {wer_conv:5.2f}%  SEQ: {wer_seq:5.2f}%  "
        f"BEST={best_label} ({best_wer:.2f}%)",
        f"{work_dir}/{mode}.txt",
    )
    return best_wer

"""
Hybrid BiLSTM-Transformer Backend
Combines the sequential modeling power of BiLSTM with the long-range attention of Transformers.

Why Hybrid > Replacement:
- Keeps BiLSTM for local sequential patterns (it works!)
- Adds Transformer for long-range dependencies and parallel processing
- Addresses "limited generalization" weakness from the baseline
- More robust than full replacement

Architecture:
    Input (T, B, C) -> BiLSTM -> LayerNorm -> Transformer -> Output (T, B, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor (seq_len, batch, d_model)
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class HybridBackend(nn.Module):
    """
    Hybrid BiLSTM-Transformer Backend for Sign Language Recognition.

    Combines:
    - BiLSTM: Captures local sequential dependencies (proven to work)
    - Transformer: Captures long-range dependencies and global context
    - Residual connections: Ensures stable training

    Args:
        rnn_type: Type of RNN (default: "LSTM")
        input_size: Input feature dimension
        hidden_size: Hidden dimension for both BiLSTM and Transformer
        num_layers: Number of BiLSTM layers
        bidirectional: Whether BiLSTM is bidirectional
        num_transformer_layers: Number of Transformer encoder layers (default: 2)
        num_heads: Number of attention heads in Transformer (default: 8)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        rnn_type="LSTM",
        input_size=512,
        hidden_size=512,
        num_layers=2,
        bidirectional=True,
        num_transformer_layers=2,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # BiLSTM component
        rnn_hidden_size = hidden_size // self.num_directions
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                rnn_hidden_size,
                num_layers,
                batch_first=False,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                rnn_hidden_size,
                num_layers,
                batch_first=False,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Layer normalization after BiLSTM
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        # Positional encoding for Transformer
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # (T, B, C) format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Layer normalization after Transformer
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

        print(
            f"✨ HybridBackend: BiLSTM({num_layers}L) + Transformer({num_transformer_layers}L, {num_heads}H)"
        )

    def _init_weights(self):
        # Initialize linear layers with proper gain (was 0.01 - BUG FIXED!)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize LayerNorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # RNN weights initialization
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

    def forward(self, src, lengths=None):
        """
        Args:
            src: Input tensor (T, B, C)
            lengths: Sequence lengths for each batch element (optional)

        Returns:
            dict with "predictions" (T, B, C) and "feat_len" (B,)
        """
        # Assume (T, B, C) as per SLR convention
        T, B, C = src.shape

        # === BiLSTM Component ===
        if lengths is not None:
            # Convert lengths to CPU for pack_padded_sequence
            if isinstance(lengths, torch.Tensor):
                lengths_cpu = lengths.cpu().int()
            else:
                lengths_cpu = torch.tensor(lengths).int()

            # Pack padded sequence for efficient processing
            packed_input = nn.utils.rnn.pack_padded_sequence(
                src, lengths_cpu, enforce_sorted=False
            )
            packed_output, _ = self.rnn(packed_input)
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, total_length=T
            )
        else:
            rnn_output, _ = self.rnn(src)

        # Layer norm + residual connection
        rnn_output = self.layer_norm1(rnn_output)
        if C == self.hidden_size:
            rnn_output = rnn_output + src  # Residual connection
        rnn_output = self.dropout(rnn_output)

        # === Transformer Component ===
        # Add positional encoding
        transformer_input = self.pos_encoder(rnn_output)

        # Create attention mask for padding if lengths provided
        if lengths is not None:
            # Create key padding mask (True for padding positions)
            # Shape: (B, T)
            mask_idx = torch.arange(T, device=src.device).unsqueeze(0).expand(B, -1)
            if isinstance(lengths, torch.Tensor):
                lengths_expanded = lengths.to(src.device).unsqueeze(1)
            else:
                lengths_expanded = torch.tensor(lengths, device=src.device).unsqueeze(1)
            key_padding_mask = mask_idx >= lengths_expanded
        else:
            key_padding_mask = None

        # Apply Transformer encoder
        transformer_output = self.transformer_encoder(
            transformer_input, src_key_padding_mask=key_padding_mask
        )

        # Layer norm + residual connection
        transformer_output = self.layer_norm2(transformer_output)
        transformer_output = transformer_output + rnn_output  # Residual from BiLSTM
        output = self.dropout(transformer_output)

        # Compute output lengths (same as input after BiLSTM)
        if lengths is not None:
            feat_len = lengths
        else:
            feat_len = torch.full((B,), T, dtype=torch.long, device=src.device)

        return {"predictions": output, "feat_len": feat_len}


# Backward compatibility with BiLSTMLayer interface
class BiLSTMLayer(HybridBackend):
    """
    Drop-in replacement for the original BiLSTMLayer with Hybrid backend.
    Maintains the same interface for seamless integration.
    """

    def __init__(
        self,
        rnn_type="LSTM",
        input_size=512,
        hidden_size=512,
        num_layers=2,
        bidirectional=True,
        num_transformer_layers=2,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

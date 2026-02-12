MODEL_CONFIG = {
    "num_sectors": 13,  # v3: 20개 → 13개 병합 (0~12)
    "linear_transformer": {
        "input_dim": 28,  # 25 (OHLCV+indicators+sentiment) + 3 (temporal: weekday/month/day)
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.2,
        "sequence_length": 60,
    },
    "cnn": {
        "input_channels": 3,
        "conv_channels": [32, 64, 128, 256],
        "output_dim": 128,
        "dropout": 0.3,
    },
    "fusion": {
        "shared_dim": 128,
        "sector_head_hidden": 64,
        "num_classes": 3,
        "dropout": 0.3,
    },
}

TRAINING_CONFIG = {
    "batch_size": 64,  # Multi-GPU에서 128 (64 * 2 GPU)
    "gradient_accumulation_steps": 2,  # effective_batch = 128 * 2 = 256
    "use_amp": True,  # Mixed Precision Training (메모리 40-50% 절감)
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
}

THREE_PHASE_CONFIG = {
    "phase1_backbone": {
        "epochs": 35,
        "lr": 1e-4,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "patience": 7,
    },
    "phase2_sector_heads": {
        "epochs_per_sector": 10,
        "lr": 5e-4,
        "scheduler": "step",
        "step_size": 5,
        "gamma": 0.5,
        "min_samples": 10000,
    },
    "phase3_finetune": {
        "epochs": 8,
        "lr": 1e-5,
        "scheduler": "cosine",
        "patience": 7,
    },
}

MARKET_SENTIMENT_CONFIG = {
    "lookback_period": 20,
    "default_values": {
        "volume_ratio": 0.33,
        "volatility_ratio": 0.33,
        "gap_direction": 0.0,
        "rsi_norm": 0.5,
        "foreign_inst_flow": 0.0,
    },
}

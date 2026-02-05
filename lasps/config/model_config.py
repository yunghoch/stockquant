MODEL_CONFIG = {
    "num_sectors": 20,
    "linear_transformer": {
        "input_dim": 25,
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
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
}

THREE_PHASE_CONFIG = {
    "phase1_backbone": {
        "epochs": 30,
        "lr": 1e-4,
        "scheduler": "cosine",
        "warmup_epochs": 5,
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
        "epochs": 5,
        "lr": 1e-5,
        "scheduler": "cosine",
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

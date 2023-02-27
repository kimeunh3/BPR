import argparse

from utils import read_json, set_seed
from preprocess import Preprocess

import torch

def main(config):
    preprocess = Preprocess(config)
    data = preprocess.load_train_data()
        
    model = models.get_models(config)

    train_set = BaseDataset(data, train_idx, config)
    val_set = BaseDataset(data, val_idx, config)

    train, valid = get_loader(train_set, val_set, config["data_loader"]["args"])

    trainer = BaseTrainer(
        model=model,
        train_data_loader=train,
        valid_data_loader=valid,
        config=config,
        fold=fold + 1,
    )

    trainer.train()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="BPR")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help='config 파일 경로 (default: "./config.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["trainer"]["seed"])

    main(config)
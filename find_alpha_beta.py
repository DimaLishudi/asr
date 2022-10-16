import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

from hw_asr.metric.utils import calc_cer, calc_wer

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


# similar to test.py, but runs shallow fusion beam search for different alpha, beta on validation
# print results per pair


def main(config, alphas, betas):
    logger = config.get_logger("hyperparameters selection")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # text_encoder
    text_encoder = config.get_text_encoder()
    ctc_decoder = config.get_ctc_decoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    beam_search_wers = defaultdict(float)
    count = 0

    with torch.inference_mode():
        for batch_num, batch in enumerate(tqdm(dataloaders["val"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            log_probs = torch.log_softmax(batch["logits"], dim=-1).detach().cpu().numpy()
            log_probs_length = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            count += len(batch["text"])
            for alpha in alphas:
                for beta in betas:
                    ctc_decoder.reset_params(alpha=alpha, beta=beta)
                    for i in range(len(batch["text"])):
                        best_hypo = ctc_decoder.decode_beams(log_probs[i][:log_probs_length[i]], beam_width=100)[0]
                        beam_search_wers[(alpha, beta)] += calc_wer(batch["text"][i], best_hypo[0])

    for key in beam_search_wers:
        beam_search_wers[key] *= 100 / count

    print("Full results:\n", beam_search_wers)
    print("\n\n" + "-"*80 + '\n\n')
    sorted_wers = sorted(beam_search_wers.items(), key=lambda item: item[1])
    print("Optimal parameters:", sorted_wers[0][0])
    print("Best wer:", sorted_wers[0][1])


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-v",
        "--val-data-folder",
        default=None,
        type=str,
        help="Path to validation dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Validation dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for valiadtion dataloader",
    )
    args.add_argument(
        "--alphas",
        default=None,
        type=float,
        nargs="+",
        help="list of possible alpha hyperparameters",
    )
    args.add_argument(
        "--betas",
        default=None,
        type=float,
        nargs="+",
        help="list of possible beta hyperparameters",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--val-data-folder` was provided, set it as a default test set
    if args.val_data_folder is not None:
        val_data_folder = Path(args.val_data_folder).absolute().resolve()
        assert val_data_folder.exists()
        config.config["data"] = {
            "val": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(val_data_folder / "audio"),
                            "transcription_dir": str(
                                val_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("val", None) is not None
    config["data"]["val"]["batch_size"] = args.batch_size
    config["data"]["val"]["n_jobs"] = args.jobs

    main(config, args.alphas, args.betas)

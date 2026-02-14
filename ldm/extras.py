import logging
from pathlib import Path
from contextlib import contextmanager

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

logger = logging.getLogger(__name__)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """A context manager that will prevent any logging messages
    triggered during the body from being processed.

    https://gist.github.com/simon-weber/7853144
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


def load_training_dir(train_dir, device, epoch="last"):
    """Load a checkpoint and config from training directory."""
    train_dir = Path(train_dir)
    ckpt = list(train_dir.rglob(f"*{epoch}.ckpt"))
    assert len(ckpt) == 1, f"found {len(ckpt)} matching ckpt files"
    config = list(train_dir.rglob(f"*-project.yaml"))
    assert len(ckpt) > 0, f"didn't find any config in {train_dir}"
    if len(config) > 1:
        logger.warning(f"found {len(config)} matching config files")
        config = sorted(config)[-1]
        logger.info(f"selecting {config}")
    else:
        config = config[0]

    config = OmegaConf.load(config)
    return load_model_from_config(config, ckpt[0], device)


def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt.
    If config is a path will use omegaconf to load.
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    with all_logging_disabled():
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        global_step = pl_sd["global_step"]
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            logger.info(f"missing keys: {m}")
        if len(u) > 0 and verbose:
            logger.info(f"unexpected keys: {u}")
        model.to(device)
        model.eval()
        model.cond_stage_model.device = device
        return model

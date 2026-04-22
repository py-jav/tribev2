# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quick test run on reduced data and number of epochs for CI."""

import os
import warnings
import logging

warnings.filterwarnings(
    "ignore", message=".*LabelEncoder: event_types has not been set.*"
)
logging.getLogger("neuralset.extractors.base").setLevel(logging.ERROR)

from exca import ConfDict

from ..main import TribeExperiment  # type: ignore
from .configs import mini_config
import torch.multiprocessing as mp


update = {
    "data.num_workers": 4,
    "data.batch_size": 32,
    "infra.cluster": None,
    "infra.workdir": None,
    "wandb_config": {"project": "JapanEEG_training"},
    "save_checkpoints": True,
    "n_epochs": 50,
    "infra.gpus_per_node": 1,
    "infra.mode": "cached",
    "data.study.infra_timelines.mode": "cached",
    "data.duration_trs": 1200,
    "data.study.names": "JapanEEG",
    "data.study.transforms.query.query": "subject_timeline_index<50",
    "data.study.transforms.split.val_ratio": 0.5,
    "data.study.infra_timelines.cluster": "processpool",
    "data.study.infra_timelines.max_jobs": 8,
    "data.study.infra_timelines.min_samples_per_job": 4,
    "data.audio_feature.infra.max_jobs": 1,
    "data.text_feature.infra.max_jobs": 1,
    "data.video_feature.infra.max_jobs": 1,
    "data.image_feature.infra.max_jobs": 1,
}

updated_config = ConfDict(mini_config)
updated_config.update(update)


def test_run(config: dict) -> None:
    task = TribeExperiment(**config)
    task.infra.clear_job()
    task.run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    folder = os.path.join(updated_config["infra"]["folder"], "test")
    updated_config["infra"]["folder"] = folder
    if os.path.exists(folder):
        import shutil

        shutil.rmtree(folder)
    test_run(updated_config)

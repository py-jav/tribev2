# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Japan EEG: EEG responses to speech stimuli (OpenNeuro ds007630).

This study provides EEG data recorded during a speech perception/decoding task.
The dataset is distributed in BIDS format on OpenNeuro (ds007630).

Experimental Design:
    - EEG recordings during a speech brain decoding task
    - BIDS-compliant format (ds007630 on OpenNeuro)
    - Subject IDs, session count, and task names must be confirmed after download

Data Format:
    - BIDS EEG format (.vhdr / .eeg / .vmrk — BrainVision, or .set — EEGLAB)
    - Stimulus event information stored in *_events.tsv per BIDS convention
    - Sampling frequency: ~1200 Hz (verify from dataset_description.json)
    - Number of channels: ~128 (verify from channels.tsv)

Download Instructions:
    - Use AWS S3 (recommended, no authentication required):
        aws s3 sync --no-sign-request s3://openneuro.org/ds007630 ./ds007630
    - Or use DataLad:
        datalad install https://github.com/OpenNeuroDatasets/ds007630
        cd ds007630 && datalad get .

Note:
    After downloading, update _SUBJECTS, _SAMPLING_FREQ, and _N_CHANNELS
    by inspecting the dataset_description.json and channels.tsv files.
    Run `--dryrun` first to preview the file list before downloading:
        aws s3 sync --no-sign-request s3://openneuro.org/ds007630 ./ds007630 --dryrun
"""

import logging
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from neuralset.events import study

logger = logging.getLogger(__name__)


class JapanEEG(study.Study):
    # Device type: EEG
    device: tp.ClassVar[str] = "Eeg"
    dataset_name: tp.ClassVar[str] = "Japan EEG Speech Decoding"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds007630"
    licence: tp.ClassVar[str] = "CC0"
    bibtex: tp.ClassVar[
        str
    ] = """
    @dataset{ds007630,
        title   = {EEG-Speech Brain Decoding Dataset},
        author  = {},
        year    = {},
        publisher = {OpenNeuro},
        doi     = {},
        url     = {https://openneuro.org/datasets/ds007630}
    }
    """
    description: tp.ClassVar[str] = (
        "EEG recordings from participants during a speech perception/decoding task. "
        "Distributed via OpenNeuro (ds007630) in BIDS format. "
        "Licensed under CC0 for unrestricted use including commercial applications."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("mne>=1.6.0",)

    # --- Dataset constants (update after downloading and inspecting the data) ---
    # Subject IDs found under ds007630/sub-*/
    _SUBJECTS: tp.ClassVar[list[str]] = []  # e.g. ["sub-P01", "sub-P02", ...]
    # Session labels found under each subject directory
    _SESSIONS: tp.ClassVar[list[str]] = ["ses-01"]
    # EEG sampling frequency in Hz (check dataset_description.json or *_eeg.json)
    _SAMPLING_FREQ: tp.ClassVar[float] = 1200.0
    # Number of EEG channels (check channels.tsv)
    _N_CHANNELS: tp.ClassVar[int] = 128
    # Notch filter frequency for Japanese power line noise
    _POWERLINE_FREQ: tp.ClassVar[float] = 50.0
    # Bandpass filter range in Hz
    _HIGHPASS_HZ: tp.ClassVar[float] = 0.1
    _LOWPASS_HZ: tp.ClassVar[float] = 500.0

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1974,  # Update after inspecting the dataset
        num_subjects=3,  # Update after inspecting the dataset
        num_events_in_query=0,  # Update after inspecting the dataset
        event_types_in_query={"Eeg", "Audio", "Word"},
        data_shape=(_N_CHANNELS, 0),  # (n_channels, n_timepoints); update n_timepoints
        frequency=_SAMPLING_FREQ,
        fmri_spaces=(),  # Not applicable for EEG
    )

    def _download(self) -> None:
        raise NotImplementedError(
            "Automatic download is not implemented. "
            "Please download manually using one of the following methods:\n"
            "  AWS S3 (recommended):\n"
            "    aws s3 sync --no-sign-request s3://openneuro.org/ds007630 ./ds007630\n"
            "  DataLad:\n"
            "    datalad install https://github.com/OpenNeuroDatasets/ds007630\n"
            "    cd ds007630 && datalad get ."
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Yield one timeline dict per EEG recording file found in the dataset.

        Scans the BIDS directory for all *_eeg.vhdr files and extracts
        subject, session, task, and run identifiers from the filename.
        Falls back to .set files (EEGLAB format) if no .vhdr files are found.
        """
        base = self.path

        # Walk directory manually instead of rglob to reduce memory usage
        for subject_dir in sorted(base.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            for session_dir in sorted(subject_dir.glob("ses-*")):
                eeg_dir = session_dir / "eeg"
                if not eeg_dir.exists():
                    continue
                for eeg_file in sorted(eeg_dir.glob("*_eeg.edf")):
                    parts: dict[str, str] = {}
                    for segment in eeg_file.stem.split("_"):
                        if "-" in segment:
                            key, value = segment.split("-", 1)
                            parts[key] = value

                    subject = (
                        f"sub-{parts['sub']}" if "sub" in parts else subject_dir.name
                    )
                    session = parts.get("ses", "01")
                    task = parts.get("task", "unknown")
                    run = parts.get("run", "01")
                    split = "test" if run == "01" else "train"

                    yield dict(
                        subject=subject,
                        session=session,
                        task=task,
                        run=run,
                        split=split,
                        eeg_path=str(eeg_file),
                    )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_eeg(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """Return an MNE Raw object with header only.

        Channel selection and filtering are handled by EegExtractor at training time.
        Only the file header is read here to keep scanning fast.
        """
        import mne

        raw = mne.io.read_raw_edf(
            str(Path(timeline["eeg_path"])), preload=False, verbose=False
        )
        return raw  # Header only — no channel selection, no filtering, no data load

    def _get_events_filepath(self, timeline: dict[str, tp.Any]) -> Path:
        """Return the BIDS *_events.tsv path corresponding to the given EEG file."""
        eeg_path = Path(timeline["eeg_path"])
        # Replace the _eeg.vhdr or _eeg.set suffix with _events.tsv
        stem = eeg_path.stem.replace("_eeg", "")
        return eeg_path.parent / f"{stem}_events.tsv"

    def _get_audio_filepath(self, timeline: dict[str, tp.Any]) -> Path | None:
        """Return the audio stimulus file path corresponding to the given EEG recording.

        Audio files are stored in the beh/ directory alongside the EEG files,
        with the naming pattern: sub-XX_ses-XX_task-XX_acq-pangolin_run-XX_recording-vocal_beh.wav
        """
        eeg_path = Path(timeline["eeg_path"])
        # Navigate from eeg/ to beh/ directory
        beh_dir = eeg_path.parent.parent / "beh"

        # Build the corresponding audio filename from the EEG filename
        # e.g. sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_eeg.edf
        #   -> sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_recording-vocal_beh.wav
        audio_stem = eeg_path.stem.replace("_eeg", "_recording-vocal_beh")
        audio_path = beh_dir / f"{audio_stem}.wav"

        if audio_path.exists():
            return audio_path

        logger.warning("Audio file not found: %s", audio_path)
        return None

    def _get_split(self, timeline: dict[str, tp.Any]) -> str:
        """Return 'train' or 'test' for the given timeline.

        Default logic: run-01 is held out as test, all others are train.
        Adjust this method once the actual train/test split is known.
        """
        return "test" if timeline.get("run") == "01" else "train"

    # ------------------------------------------------------------------
    # Required study interface
    # ------------------------------------------------------------------

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Build the events DataFrame for a single EEG recording session.

        Returns a DataFrame with one row per event, including:
        - An 'Eeg' row pointing to the raw EEG data loader
        - Optional 'Audio' row pointing to the stimulus .wav file
        - 'Word' rows for each trial onset from the BIDS *_events.tsv
        """
        all_events: list[dict[str, tp.Any]] = []

        # --- EEG data event ---
        loader_info = study.SpecialLoader(
            method=self._load_eeg, timeline=timeline
        ).to_json()
        all_events.append(
            dict(
                type="Eeg",
                filepath=loader_info,
                start=0.0,
                frequency=self._SAMPLING_FREQ,
            )
        )

        # --- Audio stimulus event (optional) ---
        audio_path = self._get_audio_filepath(timeline)
        if audio_path is not None and audio_path.exists():
            all_events.append(
                dict(
                    type="Audio",
                    filepath=str(audio_path),
                    start=0.0,
                    language="japanese",
                )
            )
        else:
            logger.warning(
                "No audio stimulus file found for task=%s; skipping Audio event.",
                timeline.get("task"),
            )

        # --- Trial/word events from BIDS events.tsv ---
        events_path = self._get_events_filepath(timeline)
        if events_path.exists():
            bids_events = pd.read_csv(events_path, sep="\t")
            for _, row in bids_events.iterrows():
                onset = float(row.get("onset", 0.0))
                duration = float(row.get("duration", 0.0))
                trial_type = str(row.get("trial_type", ""))

                # Skip non-stimulus markers (e.g. boundary, response)
                if trial_type.lower() in {"boundary", "response", "nan", ""}:
                    continue

                all_events.append(
                    dict(
                        type="Word",
                        start=onset,
                        duration=duration,
                        stop=onset + duration,
                        text=trial_type,
                        language="japanese",
                    )
                )
        else:
            logger.warning(
                "No events.tsv found at %s; only EEG event will be included.",
                events_path,
            )

        out = pd.DataFrame(all_events)

        # Mark speech events as auditory modality (consistent with other studies)
        out.loc[out["type"].isin(["Word", "Audio"]), "modality"] = "heard"

        return out

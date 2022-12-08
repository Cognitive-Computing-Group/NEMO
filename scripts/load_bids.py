"""
Converts the BIDS data to MNE objects in specified formats.
"""
import argparse
import logging

from nemo.utils import get_all_subjects
from nemo.read_bids import bids_to_mne

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--subjects",
        nargs="+",
        default=get_all_subjects(),
        help="Subject(s) to process (default: all)",
    )
    parser.add_argument(
        "-e",
        "--include_events",
        nargs="+",
        default=["empe", "afim"],
        help="Event(s) to include (default: E, R)",
    )
    parser.add_argument("--save_od", action="store_true")
    parser.add_argument("--save_haemo", action="store_true")
    parser.add_argument("--no_save_epochs", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%D %H:%M:%S",
    )
    bids_to_mne(
        subjects=args.subjects,
        include_events=args.include_events,
        save_od=args.save_od,
        save_haemo=args.save_haemo,
        save_epochs=not args.no_save_epochs,
    )

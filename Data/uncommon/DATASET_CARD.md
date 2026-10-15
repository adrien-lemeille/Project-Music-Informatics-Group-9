# Dataset Card

English | [日本語](DATASET_CARD.ja.md)

## Summary

The `uncommon-meter-beat-dataset` repository is an annotation-only dataset for beat/downbeat and time-signature analysis with an emphasis on odd meter and meter changes. It is intended to complement public beat-tracking datasets that are heavily skewed toward 4/4 music.

Current repository snapshot. This block is auto-generated from [`parsed_beats/`](parsed_beats) by `python update_dataset_docs.py`.

<!-- DATASET_CARD_STATS:START -->
- 114 tracks
- 14,148 annotated measures
- About 420.4 minutes of annotated duration
- 17 distinct time signatures
- 81 tracks with meter changes
- 95 tracks containing odd meters
<!-- DATASET_CARD_STATS:END -->

## Motivation

Many published beat-tracking benchmarks contain relatively little odd meter or changing meter material. As a result, models can appear strong on standard benchmarks while still failing on 5/4, 7/8, mixed-meter passages, and similar cases. This dataset was collected to make those cases easier to analyze, evaluate, and augment.

## What Is Included

- Original `.beat` annotation files in [`beats/`](beats)
- Normalized JSON annotations in [`parsed_beats/`](parsed_beats)
- Dataset-level statistics plots in [`output_graphs/`](output_graphs)
- Track-level metadata in [`metadata.csv`](metadata.csv)

## What Is Not Included

- No audio files
- No automatic download script for source audio
- No predefined train/validation/test split

Tracks are identified by `youtube_id`, and users are expected to obtain source audio separately while respecting the relevant rights and platform terms.

## Annotation Schema

Each track is represented as a list of measure-level annotations. Each measure stores:

- downbeat time in seconds
- time signature numerator and denominator
- local tempo in BPM
- tempo base note
- optional measure-level annotation labels

The normalized JSON format is intended to make downstream analysis easier than working directly with the original `.beat` files.

## Collection Strategy

This is a targeted dataset, not a random sample. Tracks were intentionally chosen because they contain:

- odd meter such as 5/4, 5/8, 7/4, 7/8, or 9/8
- meter changes within a track
- less common simple or compound meters that are underrepresented in many public datasets

This bias is deliberate and should be treated as a feature for evaluation and targeted augmentation, not as a claim of broad musical coverage.

## Intended Uses

- Evaluation of beat/downbeat models on odd and changing meter
- Error analysis for beat-tracking systems outside common meter
- Auxiliary or fine-tuning data for models that underperform on uncommon meter
- Time-signature distribution studies on curated non-standard-meter material

## Known Limitations

- The dataset is small relative to large-scale audio corpora.
- The meter distribution is intentionally skewed and should not be treated as natural prevalence.
- Duration statistics are approximate at the final measure of each track.
- Titles remain in filenames for readability, so `youtube_id` should be used as the canonical key.

## Legal and Licensing Notes

The repository distributes annotations and metadata only. Audio is excluded by design. The repository is released under the [MIT License](LICENSE).

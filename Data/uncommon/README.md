---
license: mit
tags:
- music
- beat
---
# uncommon-meter-beat-dataset

English | [日本語](README.ja.md)

The `uncommon-meter-beat-dataset` repository is an annotation-only beat/downbeat dataset designed to improve coverage of odd and changing meter. Public beat-tracking datasets are often dominated by 4/4 material, which makes it harder to train and evaluate models on less common meters. This repository focuses on tracks that contain uncommon time signatures, meter changes, or both.

Audio is intentionally not distributed. The repository publishes annotation files and metadata only. `youtube_id` is the stable key for each track, and [`metadata.csv`](metadata.csv) provides the lookup table.

## Current Snapshot

This block is auto-generated from [`parsed_beats/`](parsed_beats) by `python update_dataset_docs.py`.

<!-- DATASET_STATS:START -->
- 114 tracks
- 14,148 annotated measures
- About 420.4 minutes of annotated duration
- 17 distinct time signatures
- 81 tracks with meter changes
- 95 tracks containing odd meters
- 4/4 accounts for about 17.6% of all measures
- 54.9% of measures are outside the common set `{2/4, 3/4, 4/4, 6/8}`
<!-- DATASET_STATS:END -->

## Why This Dataset Exists

The goal is not to represent everyday music uniformly. The goal is to cover a failure case: beat-tracking models often generalize poorly to odd meter and changing meter because these cases are sparse in many public datasets. This collection is intentionally biased toward that gap.

## Repository Layout

```text
beats/                  Original .beat annotation files
parsed_beats/           Normalized JSON annotations used by the analysis scripts
output_graphs/          Dataset statistics figures
export_metadata_csv.py  Generates metadata.csv from the annotation files
update_dataset_docs.py  Updates snapshot sections in README and dataset cards
metadata.csv            Track-level metadata keyed by youtube_id
```

## Annotation Format

Each parsed JSON file contains:

- `youtube_id`: stable track identifier
- `measures`: list of measure-level annotations

Each measure entry includes:

- `measure_index`
- `downbeat_sec`
- `time_sig_num`
- `time_sig_den`
- `tempo_bpm`
- `base_note`
- `annotations`

This makes the repository best described as an annotation-only downbeat and time-signature dataset. Audio retrieval and any licensing checks for source media are left to the user.

## Metadata

[`metadata.csv`](metadata.csv) is generated from [`parsed_beats/`](parsed_beats) and includes:

- `youtube_id` and `youtube_url`
- track title inferred from the original filename
- original `.beat` filename and parsed JSON filename
- measure count and estimated track duration
- unique time signatures per track
- flags for meter changes, odd meters, and uncommon meters
- annotation counts and annotation label inventory

Regenerate it with:

```bash
python export_metadata_csv.py
```

## Statistics

The figures in [`output_graphs/`](output_graphs) summarize the current dataset snapshot.

![Time signature distribution by measures](output_graphs/time_signatures_distribution.png)
![Time signature distribution by tracks](output_graphs/time_signatures_distribution_tracks.png)
![Time signature distribution by duration](output_graphs/time_signatures_duration_distribution.png)

## Reproducing the Derived Files

```bash
python parse_beats.py
python analyze_beat_data.py
python export_metadata_csv.py
python update_dataset_docs.py
```

`analyze_beat_data.py` requires `matplotlib`. The metadata export script and the document snapshot updater use only the Python standard library.

## Limitations

- The collection is intentionally biased toward uncommon and changing meter, so it is not representative of general music.
- Audio is not included.
- Estimated track duration is derived from consecutive downbeats, with the final measure duration estimated from local tempo and time signature.
- Filenames keep their original human-readable titles; `youtube_id` should be treated as the canonical identifier.

## Before Publishing

This repository is released under the [MIT License](LICENSE). Audio is not included.
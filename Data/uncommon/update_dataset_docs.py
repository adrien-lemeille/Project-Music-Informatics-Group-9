from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


COMMON_METERS = {"2/4", "3/4", "4/4", "6/8"}
ODD_METER_NUMERATORS = {5, 7, 9, 11, 13, 15}
README_STATS_START = "<!-- DATASET_STATS:START -->"
README_STATS_END = "<!-- DATASET_STATS:END -->"
CARD_STATS_START = "<!-- DATASET_CARD_STATS:START -->"
CARD_STATS_END = "<!-- DATASET_CARD_STATS:END -->"


@dataclass
class DatasetStats:
    track_count: int
    measure_count: int
    total_duration_sec: float
    unique_time_signature_count: int
    meter_change_track_count: int
    odd_meter_track_count: int
    four_four_measure_ratio: float
    uncommon_measure_ratio: float

    @property
    def total_duration_min(self) -> float:
        return self.total_duration_sec / 60.0


def base_note_to_value(base_note: str) -> float:
    mapping = {
        "whole": 1.0,
        "half": 2.0,
        "quarter": 4.0,
        "eighth": 8.0,
        "16th": 16.0,
    }
    return mapping.get(base_note, 4.0)


def estimate_measure_duration_sec(measure: dict) -> float:
    bpm = float(measure.get("tempo_bpm", 120.0))
    if bpm <= 0:
        return 0.0

    time_sig_num = int(measure.get("time_sig_num", 4))
    time_sig_den = int(measure.get("time_sig_den", 4))
    base_note_value = base_note_to_value(measure.get("base_note", "quarter"))
    beats_in_measure = time_sig_num * (base_note_value / time_sig_den)
    return beats_in_measure * 60.0 / bpm


def is_odd_meter(time_signature: str) -> bool:
    numerator_text, _ = time_signature.split("/", maxsplit=1)
    return int(numerator_text) in ODD_METER_NUMERATORS


def compute_dataset_stats(parsed_dir: Path) -> DatasetStats:
    parsed_paths = sorted(parsed_dir.glob("*.json"))
    time_signature_counter: Counter[str] = Counter()
    track_count = len(parsed_paths)
    measure_count = 0
    total_duration_sec = 0.0
    meter_change_track_count = 0
    odd_meter_track_count = 0

    for parsed_path in parsed_paths:
        with parsed_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        measures = data.get("measures", [])
        measure_count += len(measures)

        track_time_signatures: list[str] = []
        for index, measure in enumerate(measures):
            time_signature = (
                f"{measure.get('time_sig_num', 4)}/{measure.get('time_sig_den', 4)}"
            )
            time_signature_counter[time_signature] += 1
            track_time_signatures.append(time_signature)

            if index + 1 < len(measures):
                next_measure = measures[index + 1]
                duration_sec = max(
                    float(next_measure.get("downbeat_sec", 0.0))
                    - float(measure.get("downbeat_sec", 0.0)),
                    0.0,
                )
            else:
                duration_sec = estimate_measure_duration_sec(measure)

            total_duration_sec += duration_sec

        unique_track_time_signatures = set(track_time_signatures)
        if len(unique_track_time_signatures) > 1:
            meter_change_track_count += 1
        if any(is_odd_meter(sig) for sig in unique_track_time_signatures):
            odd_meter_track_count += 1

    four_four_measure_ratio = 0.0
    uncommon_measure_ratio = 0.0
    if measure_count > 0:
        four_four_measure_ratio = time_signature_counter["4/4"] / measure_count
        uncommon_measure_total = sum(
            count
            for signature, count in time_signature_counter.items()
            if signature not in COMMON_METERS
        )
        uncommon_measure_ratio = uncommon_measure_total / measure_count

    return DatasetStats(
        track_count=track_count,
        measure_count=measure_count,
        total_duration_sec=total_duration_sec,
        unique_time_signature_count=len(time_signature_counter),
        meter_change_track_count=meter_change_track_count,
        odd_meter_track_count=odd_meter_track_count,
        four_four_measure_ratio=four_four_measure_ratio,
        uncommon_measure_ratio=uncommon_measure_ratio,
    )


def format_readme_stats_en(stats: DatasetStats) -> list[str]:
    return [
        f"- {stats.track_count} tracks",
        f"- {stats.measure_count:,} annotated measures",
        f"- About {stats.total_duration_min:.1f} minutes of annotated duration",
        f"- {stats.unique_time_signature_count} distinct time signatures",
        f"- {stats.meter_change_track_count} tracks with meter changes",
        f"- {stats.odd_meter_track_count} tracks containing odd meters",
        f"- 4/4 accounts for about {stats.four_four_measure_ratio * 100:.1f}% of all measures",
        (
            "- "
            f"{stats.uncommon_measure_ratio * 100:.1f}% of measures are outside the "
            "common set `{2/4, 3/4, 4/4, 6/8}`"
        ),
    ]


def format_readme_stats_ja(stats: DatasetStats) -> list[str]:
    return [
        f"- {stats.track_count} 曲",
        f"- {stats.measure_count:,} 小節",
        f"- 注釈対象の合計長さは約 {stats.total_duration_min:.1f} 分",
        f"- {stats.unique_time_signature_count} 種類の拍子",
        f"- {stats.meter_change_track_count} 曲に拍子変化あり",
        f"- {stats.odd_meter_track_count} 曲が奇数拍子を含む",
        f"- 全小節のうち 4/4 は約 {stats.four_four_measure_ratio * 100:.1f}%",
        (
            "- "
            f"`{{2/4, 3/4, 4/4, 6/8}}` 以外の拍子が全小節の約 "
            f"{stats.uncommon_measure_ratio * 100:.1f}%"
        ),
    ]


def format_card_stats_en(stats: DatasetStats) -> list[str]:
    return [
        f"- {stats.track_count} tracks",
        f"- {stats.measure_count:,} annotated measures",
        f"- About {stats.total_duration_min:.1f} minutes of annotated duration",
        f"- {stats.unique_time_signature_count} distinct time signatures",
        f"- {stats.meter_change_track_count} tracks with meter changes",
        f"- {stats.odd_meter_track_count} tracks containing odd meters",
    ]


def format_card_stats_ja(stats: DatasetStats) -> list[str]:
    return [
        f"- {stats.track_count} 曲",
        f"- {stats.measure_count:,} 小節",
        f"- 注釈対象の合計長さは約 {stats.total_duration_min:.1f} 分",
        f"- {stats.unique_time_signature_count} 種類の拍子",
        f"- {stats.meter_change_track_count} 曲に拍子変化あり",
        f"- {stats.odd_meter_track_count} 曲が奇数拍子を含む",
    ]


def replace_block(
    file_path: Path, start_marker: str, end_marker: str, block_lines: list[str]
) -> None:
    text = file_path.read_text(encoding="utf-8")
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise ValueError(f"Markers not found in {file_path}")

    block = start_marker + "\n" + "\n".join(block_lines) + "\n" + end_marker
    updated_text = text[:start_index] + block + text[end_index + len(end_marker) :]
    file_path.write_text(updated_text, encoding="utf-8")


def update_docs(parsed_dir: Path, repo_dir: Path) -> DatasetStats:
    stats = compute_dataset_stats(parsed_dir)

    replace_block(
        repo_dir / "README.md",
        README_STATS_START,
        README_STATS_END,
        format_readme_stats_en(stats),
    )
    replace_block(
        repo_dir / "README.ja.md",
        README_STATS_START,
        README_STATS_END,
        format_readme_stats_ja(stats),
    )
    replace_block(
        repo_dir / "DATASET_CARD.md",
        CARD_STATS_START,
        CARD_STATS_END,
        format_card_stats_en(stats),
    )
    replace_block(
        repo_dir / "DATASET_CARD.ja.md",
        CARD_STATS_START,
        CARD_STATS_END,
        format_card_stats_ja(stats),
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update dataset snapshot sections in README and dataset cards."
    )
    parser.add_argument(
        "--parsed_dir",
        type=Path,
        default=Path("parsed_beats"),
        help="Directory containing parsed beat JSON files.",
    )
    args = parser.parse_args()

    stats = update_docs(args.parsed_dir, Path("."))
    print("Updated README and dataset card snapshots.")
    print(f"Tracks: {stats.track_count}")
    print(f"Measures: {stats.measure_count}")
    print(f"Annotated duration: {stats.total_duration_min:.2f} min")


if __name__ == "__main__":
    main()

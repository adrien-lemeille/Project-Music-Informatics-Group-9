from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


COMMON_METERS = {"2/4", "3/4", "4/4", "6/8"}
ODD_METER_NUMERATORS = {5, 7, 9, 11, 13, 15}
YOUTUBE_FILENAME_PATTERN = re.compile(
    r"^(?P<title>.+)_(?P<youtube_id>[A-Za-z0-9_-]{11})\.beat(?:\.beats\.json)?$"
)


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


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def infer_title(file_name: str, youtube_id: str) -> str:
    suffixes = (
        f"_{youtube_id}.beat.beats.json",
        f"_{youtube_id}.beat",
        f"_{youtube_id}.json",
    )
    for suffix in suffixes:
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return Path(file_name).stem


def infer_title_and_youtube_id(file_name: str) -> tuple[str, str]:
    match = YOUTUBE_FILENAME_PATTERN.match(file_name)
    if match:
        return match.group("title"), match.group("youtube_id")

    stem = Path(file_name).stem
    if "_" in stem:
        title, youtube_id = stem.rsplit("_", maxsplit=1)
        return title, youtube_id

    return stem, ""


def find_matching_beat_file(beats_dir: Path, youtube_id: str) -> str:
    matches = sorted(beats_dir.glob(f"*_{youtube_id}.beat"))
    if not matches:
        return ""
    return matches[0].name


def compute_track_duration_sec(measures: list[dict]) -> float:
    total_duration = 0.0

    for index, measure in enumerate(measures):
        if index + 1 < len(measures):
            next_measure = measures[index + 1]
            duration_sec = max(
                float(next_measure.get("downbeat_sec", 0.0))
                - float(measure.get("downbeat_sec", 0.0)),
                0.0,
            )
        else:
            duration_sec = estimate_measure_duration_sec(measure)

        total_duration += duration_sec

    return total_duration


def build_row(parsed_path: Path, beats_dir: Path) -> dict[str, str]:
    with parsed_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    measures = data.get("measures", [])
    inferred_title, inferred_youtube_id = infer_title_and_youtube_id(parsed_path.name)
    youtube_id = inferred_youtube_id or data.get("youtube_id", "")
    beat_file_name = find_matching_beat_file(beats_dir, youtube_id)
    if beat_file_name:
        inferred_title, inferred_youtube_id = infer_title_and_youtube_id(beat_file_name)
        youtube_id = inferred_youtube_id or youtube_id
    title = inferred_title or infer_title(parsed_path.name, youtube_id)

    time_signatures = [
        f"{measure.get('time_sig_num', 4)}/{measure.get('time_sig_den', 4)}"
        for measure in measures
    ]
    ordered_time_signatures = ordered_unique(time_signatures)
    time_signature_counter = Counter(time_signatures)
    dominant_time_signature = ""
    if time_signature_counter:
        dominant_time_signature = time_signature_counter.most_common(1)[0][0]

    annotation_labels = ordered_unique(
        [
            annotation
            for measure in measures
            for annotation in measure.get("annotations", [])
        ]
    )
    annotation_event_count = sum(1 for measure in measures if measure.get("annotations"))
    annotation_label_count = sum(
        len(measure.get("annotations", [])) for measure in measures
    )
    duration_sec = compute_track_duration_sec(measures)

    return {
        "youtube_id": youtube_id,
        "youtube_url": f"https://www.youtube.com/watch?v={youtube_id}",
        "title": title,
        "beat_file": beat_file_name,
        "parsed_file": parsed_path.name,
        "measure_count": str(len(measures)),
        "duration_sec": f"{duration_sec:.4f}",
        "duration_min": f"{duration_sec / 60.0:.2f}",
        "time_signatures": "|".join(ordered_time_signatures),
        "primary_time_signature": dominant_time_signature,
        "time_signature_count": str(len(ordered_time_signatures)),
        "has_meter_change": str(len(ordered_time_signatures) > 1).lower(),
        "contains_non_44": str(any(sig != "4/4" for sig in ordered_time_signatures)).lower(),
        "contains_odd_meter": str(
            any(is_odd_meter(sig) for sig in ordered_time_signatures)
        ).lower(),
        "contains_uncommon_meter": str(
            any(sig not in COMMON_METERS for sig in ordered_time_signatures)
        ).lower(),
        "annotation_event_count": str(annotation_event_count),
        "annotation_label_count": str(annotation_label_count),
        "annotation_labels": "|".join(annotation_labels),
    }


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    track_count = len(rows)
    measure_total = sum(int(row["measure_count"]) for row in rows)
    duration_total_min = sum(float(row["duration_sec"]) for row in rows) / 60.0
    meter_change_tracks = sum(row["has_meter_change"] == "true" for row in rows)
    odd_meter_tracks = sum(row["contains_odd_meter"] == "true" for row in rows)
    uncommon_meter_tracks = sum(
        row["contains_uncommon_meter"] == "true" for row in rows
    )

    return {
        "track_count": str(track_count),
        "measure_total": str(measure_total),
        "duration_total_min": f"{duration_total_min:.2f}",
        "meter_change_tracks": str(meter_change_tracks),
        "odd_meter_tracks": str(odd_meter_tracks),
        "uncommon_meter_tracks": str(uncommon_meter_tracks),
    }


def export_metadata(parsed_dir: Path, beats_dir: Path, output_csv: Path) -> None:
    parsed_paths = sorted(parsed_dir.glob("*.json"))
    rows = [build_row(parsed_path, beats_dir) for parsed_path in parsed_paths]
    rows.sort(key=lambda row: row["parsed_file"])

    fieldnames = [
        "youtube_id",
        "youtube_url",
        "title",
        "beat_file",
        "parsed_file",
        "measure_count",
        "duration_sec",
        "duration_min",
        "time_signatures",
        "primary_time_signature",
        "time_signature_count",
        "has_meter_change",
        "contains_non_44",
        "contains_odd_meter",
        "contains_uncommon_meter",
        "annotation_event_count",
        "annotation_label_count",
        "annotation_labels",
    ]

    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    print(f"Wrote metadata for {summary['track_count']} tracks to: {output_csv}")
    print(f"Total measures: {summary['measure_total']}")
    print(f"Total annotated duration: {summary['duration_total_min']} min")
    print(f"Tracks with meter changes: {summary['meter_change_tracks']}")
    print(f"Tracks with odd meters: {summary['odd_meter_tracks']}")
    print(f"Tracks with uncommon meters: {summary['uncommon_meter_tracks']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export track-level metadata CSV from parsed beat annotations."
    )
    parser.add_argument(
        "--parsed_dir",
        type=Path,
        default=Path("parsed_beats"),
        help="Directory containing parsed beat JSON files.",
    )
    parser.add_argument(
        "--beats_dir",
        type=Path,
        default=Path("beats"),
        help="Directory containing original .beat files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("metadata.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    export_metadata(args.parsed_dir, args.beats_dir, args.output_csv)


if __name__ == "__main__":
    main()

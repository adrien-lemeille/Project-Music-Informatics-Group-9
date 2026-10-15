import argparse
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beat_loader import BeatDataset


class BeatDataAnalyzer:
    """
    Analyze BeatDataset contents and save simple distribution charts.
    """

    def __init__(self, dataset: BeatDataset):
        self.dataset = dataset
        plt.rcParams["font.family"] = "Meiryo"

    def _plot_distribution(
        self,
        counter: Counter,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        color: str,
        value_formatter=None,
        save_path: str | None = None,
    ) -> None:
        if not counter:
            print(f"No data found for: {title}")
            return

        sorted_items = counter.most_common()
        labels = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        fig_width = max(10, len(labels) * 0.65)
        plt.figure(figsize=(fig_width, 6))
        bars = plt.bar(labels, counts, color=color)
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)

        y_offset = max(counts) * 0.01 if max(counts) > 0 else 0.1
        if value_formatter is None:
            value_formatter = self._format_count_label

        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + y_offset,
                value_formatter(yval),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"  -> Saved graph: {save_path}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def _format_count_label(value: float) -> str:
        return str(int(round(value)))

    @staticmethod
    def _format_duration_minutes_label(value: float) -> str:
        return f"{value:.1f} min"

    @staticmethod
    def _base_note_to_value(base_note: str) -> float:
        mapping = {
            "whole": 1.0,
            "half": 2.0,
            "quarter": 4.0,
            "eighth": 8.0,
            "16th": 16.0,
        }
        return mapping.get(base_note, 4.0)

    def _estimate_measure_duration_sec(self, measure: dict) -> float:
        bpm = float(measure.get("tempo_bpm", 120.0))
        if bpm <= 0:
            return 0.0

        time_sig_num = int(measure.get("time_sig_num", 4))
        time_sig_den = int(measure.get("time_sig_den", 4))
        base_note_value = self._base_note_to_value(measure.get("base_note", "quarter"))
        beats_in_measure = time_sig_num * (base_note_value / time_sig_den)
        return beats_in_measure * 60.0 / bpm

    def get_time_signature_duration_seconds(self) -> Counter:
        """
        Sum actual measure durations for each time signature.

        Duration is derived from successive downbeats. The last measure of each
        track is estimated from its tempo and time signature because no next
        downbeat exists in the parsed data.
        """
        duration_counter = Counter()

        for beat_data in self.dataset:
            measures = beat_data.measures
            if not measures:
                continue

            for index, measure in enumerate(measures):
                time_signature = (
                    f"{measure.get('time_sig_num', 4)}/{measure.get('time_sig_den', 4)}"
                )
                if index + 1 < len(measures):
                    duration_sec = measures[index + 1].get(
                        "downbeat_sec", 0.0
                    ) - measure.get("downbeat_sec", 0.0)
                else:
                    duration_sec = self._estimate_measure_duration_sec(measure)

                duration_counter[time_signature] += max(duration_sec, 0.0)

        return duration_counter

    def plot_time_signatures(self, save_path: str | None = None) -> None:
        """
        Count time signatures by measures and plot the distribution.
        """
        time_sig_counter = Counter()

        for beat_data in self.dataset:
            for _, ts_num, ts_den in beat_data.get_time_signatures():
                time_sig_counter[f"{ts_num}/{ts_den}"] += 1

        self._plot_distribution(
            time_sig_counter,
            title="Time Signature Distribution by Measures",
            xlabel="Time Signature",
            ylabel="Measures",
            color="skyblue",
            save_path=save_path,
        )

    def plot_time_signatures_by_tracks(self, save_path: str | None = None) -> None:
        """
        Count time signatures by tracks.

        If one track contains the same time signature multiple times, it is
        counted once for that time signature.
        """
        time_sig_counter = Counter()

        for beat_data in self.dataset:
            track_time_signatures = {
                f"{ts_num}/{ts_den}"
                for _, ts_num, ts_den in beat_data.get_time_signatures()
            }
            for time_signature in track_time_signatures:
                time_sig_counter[time_signature] += 1

        self._plot_distribution(
            time_sig_counter,
            title="Time Signature Distribution by Tracks",
            xlabel="Time Signature",
            ylabel="Tracks",
            color="mediumseagreen",
            save_path=save_path,
        )

    def plot_time_signature_durations(self, save_path: str | None = None) -> None:
        """
        Sum actual duration for each time signature and plot the distribution.
        """
        duration_counter_sec = self.get_time_signature_duration_seconds()
        duration_counter_min = Counter(
            {
                time_signature: duration_sec / 60.0
                for time_signature, duration_sec in duration_counter_sec.items()
            }
        )

        self._plot_distribution(
            duration_counter_min,
            title="Time Signature Distribution by Duration",
            xlabel="Time Signature",
            ylabel="Total Duration (minutes)",
            color="cornflowerblue",
            value_formatter=self._format_duration_minutes_label,
            save_path=save_path,
        )

        print("Time signature durations:")
        for time_signature, duration_sec in duration_counter_sec.most_common():
            print(
                f"  {time_signature}: {duration_sec / 60.0:.2f} min "
                f"({duration_sec:.1f} sec)"
            )

    def plot_annotations_distribution(self, save_path: str | None = None) -> None:
        """
        Count annotation occurrences and plot the distribution.
        """
        annotation_counter = Counter()

        for beat_data in self.dataset:
            for _, anns in beat_data.get_annotations():
                for ann in anns:
                    annotation_counter[ann] += 1

        if not annotation_counter:
            print("No annotation data found.")
            return

        sorted_anns = annotation_counter.most_common()
        labels = [item[0] for item in sorted_anns]
        counts = [item[1] for item in sorted_anns]

        fig_width = max(10, len(labels) * 0.5)
        plt.figure(figsize=(fig_width, 6))

        bars = plt.bar(labels, counts, color="lightcoral")
        plt.title("Annotation Distribution", fontsize=14)
        plt.xlabel("Annotation", fontsize=12)
        plt.ylabel("Occurrences", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45, ha="right")

        y_offset = max(counts) * 0.01 if max(counts) > 0 else 0.1
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + y_offset,
                int(yval),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"  -> Saved graph: {save_path}")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze beat data and generate distribution graphs."
    )
    parser.add_argument(
        "--parsed_dir",
        type=str,
        default="parsed_beats",
        help="Path to the directory containing parsed .json files (default: 'parsed_beats')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_graphs",
        help="Path to the directory to save output graphs (default: 'output_graphs')",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from: {args.parsed_dir}")
    try:
        dataset = BeatDataset(args.parsed_dir)
        print(f"Loaded {len(dataset)} tracks.")

        analyzer = BeatDataAnalyzer(dataset)

        print("Generating graphs...")
        analyzer.plot_time_signatures(
            save_path=os.path.join(args.output_dir, "time_signatures_distribution.png")
        )
        analyzer.plot_time_signatures_by_tracks(
            save_path=os.path.join(
                args.output_dir, "time_signatures_distribution_tracks.png"
            )
        )
        analyzer.plot_time_signature_durations(
            save_path=os.path.join(
                args.output_dir, "time_signatures_duration_distribution.png"
            )
        )
        analyzer.plot_annotations_distribution(
            save_path=os.path.join(args.output_dir, "annotations_distribution.png")
        )

        print("Analysis completed.")
    except FileNotFoundError as error:
        print(error)

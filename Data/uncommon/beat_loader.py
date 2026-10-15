import os
import json
import glob
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class BeatData:
    """
    1曲分のbeat(ダウンビート等)のアノテーションデータを管理するクラス
    ファイル読み込みなどの状態変化・副作用を持たず、純粋なデータ保持と整形抽出のみを担当する。
    """

    youtube_id: str
    measures: List[Dict[str, Any]] = field(default_factory=list)
    file_path: str = ""

    @classmethod
    def from_json(cls, file_path: str) -> "BeatData":
        """
        JSONファイルからデータを読み込み、BeatDataインスタンスを生成するファクトリメソッド
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            youtube_id=data.get("youtube_id", ""),
            measures=data.get("measures", []),
            file_path=file_path,
        )

    def get_downbeat_times(self) -> List[float]:
        """
        全てのダウンビート時刻(sec)を1次元のリストとして取得する
        """
        return [m.get("downbeat_sec", 0.0) for m in self.measures]

    def get_annotations(self) -> List[Tuple[float, List[str]]]:
        """
        アノテーションが存在する小節について、(ダウンビート時刻, アノテーションリスト) のタプルを取得する
        """
        result = []
        for m in self.measures:
            anns = m.get("annotations", [])
            if anns:
                result.append((m.get("downbeat_sec", 0.0), anns))
        return result

    def get_time_signatures(self) -> List[Tuple[float, int, int]]:
        """
        各小節の拍子情報を (ダウンビート時刻, 分子, 分母) のタプルリストとして取得する
        """
        return [
            (
                m.get("downbeat_sec", 0.0),
                m.get("time_sig_num", 4),
                m.get("time_sig_den", 4),
            )
            for m in self.measures
        ]

    def __repr__(self) -> str:
        return (
            f"<BeatData(youtube_id='{self.youtube_id}', measures={len(self.measures)})>"
        )


class BeatDataset:
    """
    複数の曲のBeatDataを一括して管理するクラス
    """

    def __init__(self, json_dir: str):
        self.json_dir = json_dir
        self.data_dict: Dict[str, BeatData] = {}
        self._load_all()

    def _load_all(self):
        """指定ディレクトリの全JSONファイルを読み込む"""
        if not os.path.exists(self.json_dir):
            raise FileNotFoundError(f"ディレクトリが見つかりません: {self.json_dir}")

        json_files = glob.glob(os.path.join(self.json_dir, "*.json"))
        for fpath in json_files:
            try:
                beat_data = BeatData.from_json(fpath)
                self.data_dict[beat_data.youtube_id] = beat_data
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

    def get_by_id(self, youtube_id: str) -> BeatData:
        """youtube_idから対応するBeatDataを取得する"""
        return self.data_dict.get(youtube_id)

    def get_all_ids(self) -> List[str]:
        """読み込まれている全てのyoutube_idのリストを取得する"""
        return list(self.data_dict.keys())

    def __iter__(self):
        """BeatDataのイテレータを返す"""
        return iter(self.data_dict.values())

    def __len__(self) -> int:
        """読み込まれた曲数を返す"""
        return len(self.data_dict)

    def __repr__(self) -> str:
        return f"<BeatDataset(loaded_tracks={len(self.data_dict)})>"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and test BeatDataset.")
    parser.add_argument(
        "--json_dir",
        type=str,
        default="parsed_beats",
        help="Path to the directory containing parsed .json files (default: 'parsed_beats')",
    )
    args = parser.parse_args()

    # 簡単な動作テスト
    if os.path.exists(args.json_dir):
        dataset = BeatDataset(args.json_dir)
        print(dataset)

        # 最初の1曲で各種メソッドをテスト
        if len(dataset) > 0:
            first_id = dataset.get_all_ids()[0]
            beat = dataset.get_by_id(first_id)
            print(f"\n--- Testing data for: {first_id} ---")
            print(f"Total measures: {len(beat.measures)}")

            dbs = beat.get_downbeat_times()
            print(f"First 5 downbeats (sec): {dbs[:5]}")

            anns = beat.get_annotations()
            if anns:
                print(f"First 2 annotations: {anns[:2]}")
            else:
                print("No annotations in this track.")

            ts = beat.get_time_signatures()
            print(f"First 5 time signatures: {ts[:5]}")

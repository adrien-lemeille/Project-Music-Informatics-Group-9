"""
get_data_simple.py -- download the project data (annotations + audio).
 
HOW TO USE (Spyder or Jupyter):
  1. Install the packages once (in a terminal, or in a notebook cell with a leading "!"):
         pip install huggingface_hub yt-dlp
     and install ffmpeg (needed to convert the YouTube audio to .wav):
         Windows: winget install ffmpeg    macOS: brew install ffmpeg    Linux: sudo apt install ffmpeg
  2. Set PROJECT_DIR and the switches in the SETTINGS block below.
  3. Run the whole file (Spyder: F5; Jupyter: paste it in a cell and run it).
Re-running is safe: files that already exist are skipped.
"""
import csv, io, urllib.request, zipfile
from pathlib import Path
 
# ============================== SETTINGS (edit these) ==============================
PROJECT_DIR = Path(__file__).resolve().parent.parent   # <-- your project folder (contains src/, data/, ...)
 
DOWNLOAD_ANNOTATIONS     = True    # small files: beat/downbeat annotations of both datasets
DOWNLOAD_UNCOMMON_AUDIO  = True    # YouTube audio of the uncommon-meter dataset (114 tracks)
DOWNLOAD_HARMONIX_AUDIO  = False   # YouTube audio of Harmonix (large, mostly 4/4 -> optional control set)
 
LIMIT_unco = 5
LIMIT_harmo = 5            # only download the first N tracks of each audio set; 0 = download everything

MIN_ALIGNMENT = 0.95 # Harmonix only: skip tracks whose YouTube audio is poorly aligned with the annotations
# ===================================================================================
 
DATA = PROJECT_DIR / "Data"
HARMONIX_ZIP = "https://github.com/urinieto/harmonixset/archive/refs/heads/main.zip"
 
 
def get_uncommon_annotations():
    """Download the uncommon-meter dataset (annotations only) from HuggingFace into data/uncommon."""
    from huggingface_hub import snapshot_download
    snapshot_download("anime-song/uncommon-meter-beat-dataset", repo_type="dataset",
                      local_dir=DATA / "uncommon")
 
 
def get_harmonix_annotations():
    """Download the Harmonix GitHub repo as a zip and keep only the files we need in data/harmonix."""
    dest = DATA / "harmonix"
    (dest / "beats_and_downbeats").mkdir(parents=True, exist_ok=True)
    wanted = {"metadata.csv", "youtube_urls.csv", "youtube_alignment_scores.csv"}
    raw = urllib.request.urlopen(HARMONIX_ZIP, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            p = Path(name)
            if "/dataset/beats_and_downbeats/" in "/" + name and p.suffix == ".txt":
                (dest / "beats_and_downbeats" / p.name).write_bytes(z.read(name))
            elif p.name in wanted and p.parent.name == "dataset":
                (dest / p.name).write_bytes(z.read(name))
 
 
def download_audio(jobs, out_dir):
    """jobs = [(file_name_without_extension, youtube_url), ...]; saves mono 22.05 kHz .wav files."""
    import yt_dlp
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, url) in enumerate(jobs, 1):
        if (out_dir / f"{name}.wav").exists():            # already downloaded -> skip
            continue
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(out_dir / f"{name}.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "postprocessor_args": {"extractaudio": ["-ar", "22050", "-ac", "1"]},
            "quiet": True, "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            print(f"[{i}/{len(jobs)}] OK     {name}")
        except Exception as e:                            # video removed, blocked, ...
            print(f"[{i}/{len(jobs)}] FAILED {name}: {str(e)[:80]}")
            with open(out_dir.parent / "failed.txt", "a", encoding="utf-8") as f:
                f.write(f"{name}\t{url}\n")
 
 
def get_uncommon_audio():
    with open(DATA / "uncommon" / "metadata.csv", encoding="utf-8-sig") as f:   # utf-8-sig: file has a BOM
        rows = list(csv.DictReader(f))
    jobs = [(r["youtube_id"], r["youtube_url"]) for r in rows]
    download_audio(jobs[:LIMIT_unco] if LIMIT_unco else jobs, DATA / "uncommon" / "audio")
 
 
def get_harmonix_audio():
    base = DATA / "harmonix"
    with open(base / "youtube_alignment_scores.csv", encoding="utf-8-sig") as f:
        score = {r["File"]: float(r["score"]) for r in csv.DictReader(f)}
    with open(base / "youtube_urls.csv", encoding="utf-8-sig") as f:
        urls = {r["File"]: r["URL"] for r in csv.DictReader(f)}
    names = sorted(n for n in urls if score.get(n, 0) >= MIN_ALIGNMENT)   # keep well-aligned tracks only
    download_audio([(n, urls[n]) for n in (names[:LIMIT_harmo] if LIMIT_harmo else names)], base / "audio")
 
 
# ------------------------------------ run everything ------------------------------------
if DOWNLOAD_ANNOTATIONS:
    get_uncommon_annotations()
    get_harmonix_annotations()
    print("Annotations done.")
    
if DOWNLOAD_UNCOMMON_AUDIO:
    get_uncommon_audio()
if DOWNLOAD_HARMONIX_AUDIO:
    get_harmonix_audio()
print("Finished. Failed downloads (if any) are listed in data/<dataset>/failed.txt")
 
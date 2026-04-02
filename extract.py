#!/usr/bin/env python3
"""Video meaning extraction: produces summary and keywords as JSON."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel
import ollama


# フレーム解析にはマルチモーダルモデルを使用する。
# 最終的なJSON生成はテキスト特化モデルに委ねることで、日本語を含む
# 文字起こしテキストの処理精度を向上させている。
VISION_MODEL = "llava:latest"       # フレーム解析用（マルチモーダル）
TEXT_MODEL = "llama3.1:latest"      # サマリー・キーワード生成用（テキスト）
WHISPER_MODEL = "large-v3"
FRAME_INTERVAL = 15  # seconds between keyframes
SUMMARY_SENTENCES_MIN = 2           # サマリーの最小文数
SUMMARY_SENTENCES_MAX = 10           # サマリーの最大文数
KEYWORDS_MIN = 5                    # キーワードの最小数
KEYWORDS_MAX = 10                   # キーワードの最大数


def fmt_time(seconds: float) -> str:
    """秒数を MM:SS 形式の文字列に変換する。プロンプト内のタイムスタンプ表記に使用。"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def has_audio_stream(video_path: str) -> bool:
    """動画ファイルに音声トラックが存在するか確認する。

    画面収録や映像のみのコンテンツは音声トラックを持たない場合があり、
    そのまま ffmpeg で音声抽出しようとするとエラーになる。
    ffprobe で事前チェックすることで、音声なしの場合は文字起こしをスキップする。
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def extract_audio(video_path: str, output_path: str) -> None:
    """動画から音声を 16kHz モノラル WAV として抽出する。

    16kHz モノラルは faster-whisper が要求するフォーマット。
    stderr を捕捉して RuntimeError に変換することで、ffmpeg のエラーメッセージを
    スタックトレースから確認できるようにしている。
    """
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}")


def transcribe(audio_path: str, model_size: str, language: str | None) -> list[tuple[float, str]]:
    """音声ファイルを文字起こしし、(開始時刻, テキスト) のリストを返す。

    単純にテキストを結合せず、セグメントごとのタイムスタンプを保持する。
    これにより、後段のプロンプトで「いつ何を話しているか」という時系列の
    コンテキストをモデルに与えられる。

    faster-whisper は Apple Silicon でも MPS に対応していないため CPU で実行する。
    compute_type="int8" によりメモリ使用量と処理速度のバランスをとっている。
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    kwargs = {"language": language} if language else {}
    segments, _ = model.transcribe(audio_path, **kwargs)
    return [(seg.start, seg.text.strip()) for seg in segments]


def extract_keyframes(video_path: str, output_dir: str, interval: int) -> list[tuple[float, str]]:
    """動画から interval 秒ごとにキーフレームを抽出し、(タイムスタンプ, ファイルパス) のリストを返す。

    ffmpeg の fps=1/interval フィルタで均等間隔のフレームを抽出する。
    ファイル名のソート順（frame_001, frame_002, ...）とインデックスが対応しているため、
    i * interval でタイムスタンプを逆算できる。
    """
    pattern = os.path.join(output_dir, "frame_%03d.jpg")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps=1/{interval}", "-q:v", "2", pattern],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}")
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [(i * interval, str(f)) for i, f in enumerate(frames)]


def analyze_frame(image_path: str, model: str) -> str:
    """1枚のフレーム画像をllavaに渡し、映像内容のテキスト説明を得る。"""
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": "Describe what you see in this video frame concisely. Focus on actions, objects, people, and scene context.",
            "images": [image_path],
        }],
    )
    return response["message"]["content"]


def generate_meaning(
    transcript: list[tuple[float, str]] | None,
    timed_frames: list[tuple[float, str]],  # (timestamp, description)
    model: str = TEXT_MODEL,
) -> dict:
    """文字起こしとフレーム説明を統合し、summary と keywords を含む dict を返す。

    音声と映像それぞれに MM:SS 形式のタイムスタンプを付与してプロンプトに渡す。
    時系列が揃った状態でモデルに提示することで、「このタイミングで話していた内容が
    この映像」という対応関係を理解させ、より文脈に沿ったサマリー生成を促す。

    format="json" を指定することで ollama に JSON モードを強制し、
    パースエラーのリスクを低減している。

    音声なしの場合は映像のみのプロンプトにフォールバックする。
    """
    frames_text = "\n".join(f"[{fmt_time(t)}] {desc}" for t, desc in timed_frames)

    if transcript:
        transcript_text = "\n".join(f"[{fmt_time(t)}] {text}" for t, text in transcript)
        prompt = f"""You are analyzing a video to extract its core meaning and content.
Your goal is to understand what this video is fundamentally about — its topic, message, and what is happening visually.
Use the transcript and visual frames as evidence to reason about the video's content.
Timestamps are in MM:SS format. The transcript may be in Japanese — use the same language for your output.
# ↑ 役割の明示：テキストの要約ではなく「動画の内容理解」を目的として設定。
#   音声・映像を証拠として動画の本質を推論するよう誘導している。
#   "use the same language" は日本語の文字起こしが渡された場合に
#   モデルが英語で出力してしまうのを防ぐための指定。

TRANSCRIPT (what is being said, with timestamps):
{transcript_text}
# ↑ タイムスタンプ付きの文字起こし。faster-whisper が返すセグメント単位の
#   開始時刻をそのまま利用している。

VISUAL FRAMES (what is being shown, with timestamps):
{frames_text}
# ↑ タイムスタンプ付きのフレーム説明。キーフレームのインデックスと
#   抽出間隔から逆算したタイムスタンプを付与している。

Use the timestamps to understand how the audio and visuals relate over time.
Focus on what is actually happening in the video — the scenes, actions, and topics being presented.
# ↑ タイムスタンプを使って音声と映像の時系列上の対応関係を把握するよう促す指示。
#   「動画で実際に何が起きているか」に焦点を当てるよう明示し、
#   テキストの言い換えではなく映像コンテンツの理解を求めている。

Output a JSON object with:
- "summary": {SUMMARY_SENTENCES_MIN}-{SUMMARY_SENTENCES_MAX} sentences describing what this video is about, grounded in both what is seen and what is said
- "keywords": {KEYWORDS_MIN}-{KEYWORDS_MAX} keywords that capture the main topics and visual elements of the video
# ↑ 出力スキーマの指定。"grounded in both" で音声・映像双方への言及を求め、
#   keywords も「映像要素」を含めるよう明示している。

Output valid JSON only."""
# ↑ JSON以外のテキスト（前置き・後置き）を出力させないための指示。
#   format="json" と合わせて二重に JSON 出力を強制している。
    else:
        # 音声なし（映像のみ）の場合のフォールバックプロンプト
        prompt = f"""Summarize this video based on the visual frame descriptions below. Timestamps are in MM:SS format.

VISUAL FRAMES (with timestamps):
{frames_text}

Output a JSON object with:
- "summary": {SUMMARY_SENTENCES_MIN}-{SUMMARY_SENTENCES_MAX} sentence summary of what is shown
- "keywords": {KEYWORDS_MIN}-{KEYWORDS_MAX} relevant keywords

Output valid JSON only."""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(response["message"]["content"])


def process_video(
    video_path: str,
    vision_model: str = VISION_MODEL,
    text_model: str = TEXT_MODEL,
    frame_interval: int = FRAME_INTERVAL,
    whisper_model: str = WHISPER_MODEL,
    language: str | None = None,
) -> dict:
    """動画ファイルを受け取り、意味抽出結果を dict で返すメインパイプライン。

    中間ファイル（音声WAV・フレームJPG）は tempfile.TemporaryDirectory で管理し、
    処理完了後に自動削除される。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        if has_audio_stream(video_path):
            print("Extracting audio...")
            audio_path = os.path.join(tmpdir, "audio.wav")
            extract_audio(video_path, audio_path)
            print(f"Transcribing ({whisper_model})...")
            transcript = transcribe(audio_path, whisper_model, language)
        else:
            print("No audio track — skipping transcription.")
            transcript = None

        print(f"Extracting keyframes (every {frame_interval}s)...")
        timed_frames = extract_keyframes(video_path, tmpdir, frame_interval)
        print(f"  {len(timed_frames)} frames extracted")

        print(f"Analyzing frames ({vision_model})...")
        timed_descriptions = []
        for i, (ts, frame_path) in enumerate(timed_frames):
            print(f"  Frame {i + 1}/{len(timed_frames)} [{fmt_time(ts)}]")
            desc = analyze_frame(frame_path, vision_model)
            timed_descriptions.append((ts, desc))

        print(f"Generating summary and keywords ({text_model})...")
        return generate_meaning(transcript, timed_descriptions, text_model)


def main():
    parser = argparse.ArgumentParser(description="Extract summary and keywords from a video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", help="Output JSON file path (default: stdout)")
    parser.add_argument("--vision-model", default=VISION_MODEL, help=f"Ollama vision model for frame analysis (default: {VISION_MODEL})")
    parser.add_argument("--text-model", default=TEXT_MODEL, help=f"Ollama text model for summary/keywords (default: {TEXT_MODEL})")
    parser.add_argument("--frame-interval", type=int, default=FRAME_INTERVAL, help=f"Seconds between keyframes (default: {FRAME_INTERVAL})")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, help=f"Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--language", help="Audio language code e.g. ja, en (default: auto-detect)")
    args = parser.parse_args()

    result = process_video(
        args.video,
        vision_model=args.vision_model,
        text_model=args.text_model,
        frame_interval=args.frame_interval,
        whisper_model=args.whisper_model,
        language=args.language,
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()

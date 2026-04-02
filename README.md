動画ファイルから意味情報（サマリー・キーワード）をJSONとして抽出するローカルLLMツール。
外部APIに依存せず、すべての処理をローカルで完結する。

## 何をやるか

1. **音声抽出** — ffmpegで動画から16kHz monoのWAVを生成
2. **文字起こし** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（`large-v3`）でタイムスタンプ付きトランスクリプトを生成
3. **キーフレーム抽出** — ffmpegで15秒間隔のJPGを抽出
4. **フレーム解析** — Ollama + `llava:latest`（マルチモーダル）で各フレームを説明テキスト化
5. **JSON生成** — Ollama + `llama3.1:latest`（テキスト特化）でsummary/keywordsを生成

音声・映像にそれぞれタイムスタンプを付与してLLMに渡すことで、時系列の対応関係を考慮したサマリーを生成する。

### 出力スキーマ

```json
{
  "summary": ["動画全体の要約" ],
  "keywords": ["キーワード1", "キーワード2", "..."]
}
```

## 必要環境

- macOS（Apple Silicon推奨）
- [Homebrew](https://brew.sh/)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

## セットアップ

```bash
# 1. ffmpegのインストール
brew install ffmpeg

# 2. Ollamaのインストール（未インストールの場合）
brew install ollama

# 3. 必要なモデルのダウンロード
ollama pull llava:latest
ollama pull llama3.1:latest

# 4. Pythonパッケージのインストール
uv sync
```

## 使い方

```bash
# 結果をstdoutに出力
uv run extract.py video.mp4

# JSONファイルに保存
uv run extract.py video.mp4 -o result.json

# 日本語音声（言語を明示することで精度向上）
uv run extract.py video.mp4 --language ja -o result.json
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--vision-model` | `llava:latest` | フレーム解析に使うOllamaビジョンモデル |
| `--text-model` | `llama3.1:latest` | サマリー生成に使うOllamaテキストモデル |
| `--frame-interval` | `15` | キーフレーム抽出間隔（秒） |
| `--whisper-model` | `large-v3` | Whisperのモデルサイズ |
| `--language` | 自動検出 | 音声言語コード（例：`ja`, `en`） |

## 動作確認環境

- Python 3.14
- Ollama 0.6+

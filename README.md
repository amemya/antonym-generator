# 日本語対義語ジェネレータ (`AntonymGenerator.jl`)

Julia 1.12 と `Flux.jl` による **2 段階 Retrieve & Rerank 方式（Bi-Encoder + Cross-Encoder）** の日本語対義語候補ランキング生成システムです。

アルゴリズム仕様・損失関数・データ分割ルール・評価指標の設計詳細は [SPEC.md](SPEC.md) を参照してください。本ドキュメントでは、リポジトリ構成、Git 管理外の大容量ファイルの配置・生成方法、環境構築（Mac CPU / Windows CUDA GPU）、CLI 実行方法および Julia API をまとめます。

---

## 1. ドキュメント構成と実装上の補足

- **[SPEC.md](SPEC.md)**: 設計仕様書（目的、Stage 1 Bi-Encoder / Stage 2 Cross-Encoder / Stage 3 後処理、リーク防止データ分割、評価指標）
- **本ファイル (`README.md`)**: 実装構成、大容量データの配置・受け渡し方法、環境セットアップ、CLI / API リファレンス

### `SPEC.md` に対する実装上の補足

- **BERT バックボーンの純粋 Julia 実装**: [SPEC.md](SPEC.md) 第 5 節に記載されている `Transformers.jl` (0.3.1) は `Flux 0.14` / `Zygote 0.6` に固定されており Julia 1.12 の `Flux ~0.16` / `Zygote ~0.7` と競合するため、`Flux ~0.16` + `NNlib ~0.9` + `Zygote ~0.7` + `Optimisers ~0.4` 上にネイティブの文字単位 BERT バックボーン（[src/bert.jl](src/bert.jl)・[src/tokenizer.jl](src/tokenizer.jl)）および HuggingFace `.safetensors` ローダを実装しています。
- **使用データソース**: データ構築スクリプト（[src/data.jl](src/data.jl)）は公開辞書である **日本語 WordNet v2.1** と **Sudachi 同義語辞書 (`synonyms.txt`)** を自動取得・統合して教師データと語彙を構築します（対義語グラフの連結成分単位で `train` / `val` / `test` を完全非重複分割）。

---

## 2. ディレクトリ・ファイル構成

```text
antonym-generator/
├── .gitignore
├── Project.toml                 # Julia 1.12 パッケージ依存関係定義
├── Manifest.toml                # 解決済み依存関係ロックファイル
├── README.md                    # 本ドキュメント（環境構築・運用・リファレンス）
├── SPEC.md                      # アルゴリズム・システム設計仕様書
├── src/
│   ├── AntonymGenerator.jl      # モジュール定義・公開シンボルのエクスポート
│   ├── device.jl                # CPU / CUDA GPU 切り替え・動的ロード
│   ├── tokenizer.jl             # 文字単位 BERT トークナイザ・8次元ペア表層特徴量抽出
│   ├── bert.jl                  # Flux ネイティブ BERT 実装 & .safetensors ローダ
│   ├── models.jl                # Stage 1 BiEncoder / Stage 2 CrossEncoder / 損失関数
│   ├── data.jl                  # WordNet v2.1 & Sudachi 取得・連結成分非重複分割
│   ├── pipeline.jl              # 語彙インデックス構築・2段階推論・Stage 3 後処理
│   ├── evaluate.jl              # Recall@K / MRR / Synonym Error Rate@5 評価
│   └── train.jl                 # Stage 1 学習 / Hard Negative Mining / Stage 2 学習 / JLD2 保存・読込
├── scripts/
│   ├── prepare_data.jl          # データセットダウンロード・前処理 CLI
│   ├── train_and_eval.jl        # Stage 1〜2 学習・テスト評価・チェックポイント保存 CLI
│   └── predict.jl               # 学習済みモデルによる対義語候補生成 CLI
├── test/
│   └── runtests.jl              # 単体・結合テストスイート (`Pkg.test()`)
│
├── data/                        # [Git管理外: .gitignore 対象]
│   ├── raw/                     # 元の公開辞書アーカイブ・展開 XML・テキスト (~55MB)
│   ├── processed/               # 分割済み train/val/test/vocab CSV (~3.7MB)
│   └── pretrained/              # HuggingFace 事前学習済みモデル重み (~19MB〜数百MB)
└── artifacts/                   # [Git管理外: .gitignore 対象] 学習済み JLD2 & metrics.json (~78MB〜)
```

---

## 3. 大容量ファイル（Git 管理外）の配置・生成方法

バイナリ重みやダウンロード済み辞書データは `.gitignore` により Git 管理から除外されています。GitHub にはソースコードのみをプッシュし、以下のディレクトリは各環境で自動生成または手動配置してください。

| パス | 主なファイル | 容量目安 | 用途 | 入手・生成方法 |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/` | `wnja-2.1.tar.xz`, `wnja/wnja-2.1.xml`, `sudachi_synonyms.txt` | 約 55 MB | 元辞書データ | `scripts/prepare_data.jl` 実行で自動ダウンロード＆展開 |
| `data/processed/` | `train.csv`, `val.csv`, `test.csv`, `vocab.csv` | 約 3.7 MB | 連結成分ベース非重複データセット | `scripts/prepare_data.jl` 実行で決定論的 (`seed=42`) に生成 |
| `data/pretrained/<org>__<model>/` | `config.json`, `vocab.txt`, `model.safetensors` | 約 19 MB 〜 数百 MB | HuggingFace 事前学習 BERT | `scripts/train_and_eval.jl` 実行時に HuggingFace から自動取得 |
| `artifacts/` | `bi_encoder.jld2`, `cross_encoder.jld2`, `vocab_index.jld2`, `metrics.json` | 約 78 MB 〜 | 学習済みモデル＆語彙インデックス | `scripts/train_and_eval.jl` 実行で生成（または別マシンからコピー） |

### 3.1 自動取得・自動生成（通常手順）

インターネット接続がある環境では、手動ダウンロードなしでスクリプト実行時に自動取得されます。

1. **辞書・前処理データ (`data/raw/`, `data/processed/`)**:
   `julia --project=. scripts/prepare_data.jl` を実行すると、以下を自動ダウンロードして CSV を構築します。
   - 日本語 WordNet v2.1: `https://github.com/bond-lab/wnja/releases/download/v2.1/wnja-2.1.tar.xz`
   - Sudachi 同義語辞書: `https://raw.githubusercontent.com/WorksApplications/SudachiDict/develop/src/main/text/synonyms.txt`
2. **事前学習済み BERT (`data/pretrained/`)**:
   `scripts/train_and_eval.jl` 実行時、`--pretrained <repo_id>` で指定されたモデルの `config.json`・`vocab.txt`・`model.safetensors` を `https://huggingface.co/<repo_id>/resolve/main/` から自動取得し、`data/pretrained/<org>__<model>/`（`/` を `__` に置換したディレクトリ名）にキャッシュします。
   - 例 (`bluolightning/bert-tiny-japanese-char`): `data/pretrained/bluolightning__bert-tiny-japanese-char/`
   - 例 (`kit-nlp/bert-base-japanese-char-v3-irony`): `data/pretrained/kit-nlp__bert-base-japanese-char-v3-irony/`

### 3.2 手動配置する場合（オフライン環境・展開エラー回避）

スクリプトは配置済みファイルを検知してダウンロードをスキップします。手動配置する場合は以下のパス構造に合わせて置いてください。

* **元辞書 (`data/raw/`)**:
  * `data/raw/wnja/wnja-2.1.xml`（または `data/raw/wnja-2.1.tar.xz`）
  * `data/raw/sudachi_synonyms.txt`
  * ※ Windows 環境などで `tar -xJf` による `.tar.xz` 自動展開ができない場合は、あらかじめ `wnja-2.1.tar.xz` を展開して `data/raw/wnja/wnja-2.1.xml` を配置してください。
* **事前学習済みモデル (`data/pretrained/`)**:
  * `data/pretrained/<org>__<model>/` の直下に `config.json`, `vocab.txt`, `model.safetensors` の 3 ファイルを配置します。
  * あるいは任意のローカルフォルダに上記 3 ファイルを置き、`--pretrained /path/to/local_model_dir` のようにディレクトリパスを直接指定することもできます。

### 3.3 Windows GPU マシン ⇄ Mac 間の学習済みモデル受け渡し

`artifacts/` に保存される重みファイル（`bi_encoder.jld2`, `cross_encoder.jld2`, `vocab_index.jld2`）は、`save_artifacts` によってすべて **CPU 配列形式 (`Flux.cpu`)** でシリアライズされます。

そのため、**Windows GPU マシンで本学習して生成された `artifacts/` フォルダ一式をそのまま Mac の `antonym-generator/artifacts/` にコピーするだけで**、追加の変換なしに Mac 上で推論スクリプト (`scripts/predict.jl`) や Julia API から利用できます。

---

## 4. 環境構築手順

### 必須要件

- **Julia 1.12 以降**（[Project.toml](Project.toml) 指定: `julia = "1.12"`）

### 4.1 Mac (CPU 実行 / 開発・推論・軽量検証用)

リポジトリをクローン後、プロジェクトルートで依存パッケージをインストール・プリコンパイルします。

```bash
cd antonym-generator
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

※ macOS 上では `CUDA.jl` は不要です。`--device auto`（デフォルト）は CUDA 非搭載環境を検出し、自動的に CPU モードへフォールバックします。

### 4.2 Windows (NVIDIA CUDA GPU / 本学習用)

Windows GPU マシンでは、まず同様に `Pkg.instantiate()` を行い、加えて `CUDA.jl` を利用可能にします。

#### 方法 A: プロジェクト環境へ追加する（手軽）

```powershell
cd antonym-generator
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.add("CUDA"); using CUDA; CUDA.versioninfo()'
```

#### 方法 B: グローバル環境へ追加する（`Project.toml` / `Manifest.toml` に差分を出さない）

```powershell
cd antonym-generator
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -e 'using Pkg; Pkg.add("CUDA"); using CUDA; CUDA.versioninfo()'
```

[src/device.jl](src/device.jl) は `--device auto` または `--device cuda` 指定時に `@eval import CUDA` で動的ロードするため、上記いずれの方法でも GPU を認識できます。

---

## 5. 実行手順と CLI リファレンス

### Step 1: データセットの準備 (`scripts/prepare_data.jl`)

初回のみ実行します（すでに `data/processed/train.csv` 等がある場合は再実行不要です）。

```bash
julia --project=. scripts/prepare_data.jl
```

`train.csv` / `val.csv` / `test.csv` / `vocab.csv` が生成され、語彙数・連結成分数・`train∩val` / `train∩test` / `val∩test` の重複数が `0` であることが表示されます。

---

### Step 2: 学習とテスト評価 (`scripts/train_and_eval.jl`)

Stage 1 Bi-Encoder の学習 → Hard Negative Mining → Stage 2 Cross-Encoder の学習 → 全語彙インデックス構築 → Test セット評価 → `artifacts/` 保存を一括で実行します。

#### CLI オプション一覧

| オプション | 型 | デフォルト値 | 説明 |
| :--- | :--- | :--- | :--- |
| `--device` | String | `"auto"` | 実行デバイス（`"auto"`, `"cpu"`, `"cuda"`） |
| `--pretrained` | String | `"bluolightning/bert-tiny-japanese-char"` | HuggingFace モデル ID またはローカルディレクトリパス |
| `--epochs-stage1` | Int | `5` | Stage 1 Bi-Encoder のエポック数 |
| `--epochs-stage2` | Int | `4` | Stage 2 Cross-Encoder のエポック数 |
| `--batch-size` | Int | `64` | 学習時のミニバッチサイズ |
| `--lr-stage1` | Float32 | `3.0f-4` | Stage 1 学習率 |
| `--lr-stage2` | Float32 | `2.0f-4` | Stage 2 学習率 |
| `--proj-dim` | Int | `256` | Bi-Encoder の出力埋め込み次元数 |
| `--freeze-backbone` | Bool | `false` | 指定すると BERT バックボーンを固定し射影・分類ヘッドのみ学習（値なしのフラグ指定、または `--freeze-backbone=true`） |
| `--max-train-pairs` | Int / nothing | `nothing` | 学習に使用するペア数の上限（CPU での軽量動作確認用、`nothing` で全件使用） |
| `--max-eval-queries` | Int / nothing | `nothing` | Test 評価時に使用するクエリ単語数の上限（`nothing` で全件評価） |
| `--num-layers` | Int / nothing | `nothing` | 事前学習 BERT から読み込む Transformer 層数の上限 |
| `--top-k-stage1` | Int | `200` | Stage 1 で抽出して Stage 2 へ渡す候補数 K |
| `--data-dir` | String / nothing | `nothing` | 前処理済み CSV ディレクトリ（省略時は `data/processed`） |
| `--artifacts-dir` | String / nothing | `nothing` | 学習結果の保存先ディレクトリ（省略時は `artifacts`） |

#### 実行例 1: Mac CPU での動作確認（軽量モデル＋サブセット）

```bash
julia --project=. scripts/train_and_eval.jl \
  --device cpu \
  --pretrained bluolightning/bert-tiny-japanese-char \
  --epochs-stage1 3 \
  --epochs-stage2 3 \
  --batch-size 128 \
  --max-train-pairs 1500 \
  --max-eval-queries 50 \
  --top-k-stage1 100
```

#### 実行例 2: Windows CUDA GPU (VRAM 20GB) での 12 層 BERT 本学習

全データを使用し、12 層文字単位日本語 BERT（`kit-nlp/bert-base-japanese-char-v3-irony`）を学習する推奨コマンドです。

```powershell
julia --project=. scripts/train_and_eval.jl `
  --device cuda `
  --pretrained kit-nlp/bert-base-japanese-char-v3-irony `
  --epochs-stage1 15 `
  --epochs-stage2 8 `
  --batch-size 128 `
  --top-k-stage1 200
```

---

### Step 3: 対義語候補の推論 (`scripts/predict.jl`)

学習済みの `artifacts/` を読み込み、指定した単語に対して対義語候補をランキング出力します。位置引数として任意の数の単語を指定できます（単語を省略した場合はデフォルトの `["高い", "可能", "増加", "暑い"]` で実行されます）。

#### CLI オプション一覧

| オプション | 型 | デフォルト値 | 説明 |
| :--- | :--- | :--- | :--- |
| `--artifacts-dir` | String | `"artifacts"` | 学習済みモデルの保存ディレクトリ |
| `--top-n` | Int | `10` | 最終出力する上位候補数 |
| `--top-k-stage1` | Int | `200` | Stage 1 で絞り込む候補数 K |
| `--device` | String | `"auto"` | 実行デバイス（`"auto"`, `"cpu"`, `"cuda"`） |

#### 実行例

```bash
julia --project=. scripts/predict.jl --top-n 10 --top-k-stage1 200 高い 可能 増加 暑い
```

---

## 6. Julia コードからの API 利用例

`AntonymGenerator` モジュールを直接読み込んで推論を組み込むこともできます。

```julia
using AntonymGenerator

device_cfg = resolve_device(:auto)
pipeline = load_artifacts("artifacts"; device_cfg = device_cfg)

candidates = generate_antonyms(
    pipeline,
    "高い";
    top_k_stage1 = 200,
    top_n = 10,
)

for (rank, c) in enumerate(candidates)
    println(
        rank, ": ", c.word,
        " (", c.reading, ")",
        " | P(Antonym)=", round(c.antonym_prob; digits = 4),
        " | P(Synonym)=", round(c.synonym_prob; digits = 4),
        " | Stage1=", round(c.stage1_score; digits = 4),
    )
end
```

---

## 7. テストの実行

単体・結合テスト（トークナイザ、`.safetensors` ローダ、BERT フォワード、損失関数、連結成分分割、後処理、JLD2 保存・復元）は以下のコマンドで実行できます。

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

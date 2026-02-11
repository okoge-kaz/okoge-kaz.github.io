---
title: "CUDA行列積の最適化: 初心者からcuBLAS級の性能まで"
draft: true
date: 2026-02-08
tags: ["CUDA", "GPU", "HPC", "行列積", "最適化"]
description: "CUDA C++の基礎から始めて、行列積カーネルを段階的に最適化し、cuBLASの93%の性能に到達するまでのステップバイステップガイド。メモリコアレッシング、共有メモリ、タイリング、Tensor Coreまで網羅。"
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## はじめに

GPU プログラミングの世界に足を踏み入れたいけれど、CUDA C++ のハードルが高いと感じている方は多いだろう。本記事では **CUDA の基礎概念** から始めて、**行列積（GEMM）カーネル** を題材に段階的に最適化を進め、最終的に **cuBLAS の 93% の性能** に到達するまでの道のりを辿る。

行列積を題材にする理由は単純で、Deep Learning の学習・推論における計算時間の大部分は行列積が占めているからである。Transformer の各層で行われる Linear 演算は本質的に GEMM であり、この最適化を理解することは LLM の高速化に直結する。

本記事の構成:

1. **Level 0**: CUDA の基礎概念（スレッド、ブロック、グリッド）
2. **Level 1**: ナイーブな行列積カーネル
3. **Level 2**: メモリコアレッシング
4. **Level 3**: 共有メモリ（SMEM）によるキャッシュ
5. **Level 4**: 1D Blocktiling（スレッドあたり複数要素）
6. **Level 5**: 2D Blocktiling（外積による計算）
7. **Level 6**: ベクトル化メモリアクセス
8. **Level 7**: Warptiling
9. **Level 8**: Hopper 世代の Tensor Core（WGMMA）

> **参考文献**: 本記事は [siboehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)、[NVIDIA: An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)、[Colfax Research: CUTLASS Tutorial WGMMA](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) を参考にしている。

---

## Level 0: CUDA の基礎概念

### GPU はなぜ速いのか

CPU は少数の高性能コア（数個〜数十個）で逐次処理を高速に行う設計である。一方 GPU は、数千個の小さなコアで大量のスレッドを同時に実行する **超並列アーキテクチャ** である。

```
CPU: 少数の強いコア → 逐次処理向き
     [====] [====] [====] [====]   ← 4コア、各コアが高性能

GPU: 大量の小さなコア → 並列処理向き
     [=][=][=][=][=][=][=][=]...   ← 数千コア、各コアは単純
```

行列の各要素の計算は独立しているため、GPU の並列性と非常に相性が良い。

### CUDA のスレッド階層

CUDA では計算を **スレッド → ブロック → グリッド** という階層で組織する。これは GPU のハードウェア構造に対応している。

```
Grid（グリッド）
├── Block (0,0)          Block (1,0)          Block (2,0)
│   ├── Thread 0         ├── Thread 0         ├── Thread 0
│   ├── Thread 1         ├── Thread 1         ├── Thread 1
│   ├── ...              ├── ...              ├── ...
│   └── Thread 255       └── Thread 255       └── Thread 255
├── Block (0,1)          Block (1,1)          ...
│   └── ...              └── ...
```

- **スレッド (Thread)**: 最小の実行単位。各スレッドが独立に命令を実行する
- **ブロック (Block)**: スレッドの集まり（最大1024スレッド）。ブロック内のスレッドは **共有メモリ (Shared Memory)** を介して協調できる
- **グリッド (Grid)**: ブロックの集まり。カーネル起動時に指定する
- **ワープ (Warp)**: 32スレッドの集まり。GPU が実際にスケジューリングする単位（SIMT: Single Instruction, Multiple Threads）

### 最初の CUDA プログラム: ベクトル加算

CUDA の基本的な流れを掴むため、まずベクトル加算を実装する。

**Step 1: CPU 版**

```cpp
// CPU版: 逐次処理
void add_cpu(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}
```

**Step 2: CUDA カーネルに変換**

```cpp
// __global__ 修飾子で GPU 上で実行される関数（カーネル）を定義
__global__ void add_kernel(int n, float *x, float *y) {
    // 各スレッドが自分の担当するインデックスを計算
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // grid-stride loop: スレッド数より要素数が多くても処理できる
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}
```

> **補足: `__global__` とは？**
> `__global__` はCUDAの関数修飾子で、「この関数はCPUから呼び出されるが、GPU上で実行される」ことを意味する。これを **カーネル (kernel)** と呼ぶ。他に `__device__`（GPU上で呼び出し・実行）、`__host__`（CPU上で呼び出し・実行、デフォルト）がある。

> **補足: grid-stride loop パターン**
> `for (int i = index; i < n; i += stride)` は CUDA の定番パターンである。スレッド数が要素数より少ない場合でも、各スレッドが stride 分ずつ飛ばしながら全要素を処理できる。例えば1024スレッドで100万要素を処理する場合、各スレッドは約1000要素を担当する。

**Step 3: メモリ確保とカーネル起動**

```cpp
int main() {
    int N = 1 << 20;  // 約100万要素
    float *x, *y;

    // Unified Memory: CPU と GPU の両方からアクセス可能なメモリを確保
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // 初期化
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // カーネル起動: <<<ブロック数, ブロックあたりスレッド数>>>
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add_kernel<<<numBlocks, blockSize>>>(N, x, y);

    // GPU の処理完了を待つ
    cudaDeviceSynchronize();

    cudaFree(x);
    cudaFree(y);
    return 0;
}
```

> **補足: `<<<numBlocks, blockSize>>>` とは？**
> これは CUDA 独自のカーネル起動構文である。`<<<グリッド内のブロック数, ブロック内のスレッド数>>>` を指定する。上の例では `numBlocks × 256` 個のスレッドが並列に起動される。

> **補足: `cudaDeviceSynchronize()` の役割**
> カーネル起動は **非同期** である。つまり CPU は GPU の処理完了を待たずに次の命令に進む。`cudaDeviceSynchronize()` を呼ぶことで、GPU 上の全カーネルが完了するまで CPU を待機させる。

### GPU メモリ階層

CUDA プログラミングで最も重要な概念の一つが **メモリ階層** である。上位ほど高速だが容量が小さい。

```
レジスタ (Register)     ← 最速、スレッド固有、数KB/スレッド
    ↓
共有メモリ (SMEM)       ← 高速、ブロック内共有、48-164KB/SM
    ↓
L2 キャッシュ            ← GPU全体で共有、数MB
    ↓
グローバルメモリ (GMEM)  ← 最遅、全スレッドからアクセス可能、数十GB
```

| メモリ | 帯域幅 (概算) | レイテンシ | スコープ |
|--------|--------------|-----------|---------|
| レジスタ | - | ~1 cycle | スレッド |
| 共有メモリ | ~12 TB/s | ~20 cycles | ブロック |
| L2 キャッシュ | ~2 TB/s | ~200 cycles | GPU 全体 |
| グローバルメモリ (HBM) | ~0.8-3.3 TB/s | ~400 cycles | GPU 全体 |

この階層を意識して、**データをできるだけ高速なメモリに置いて再利用する** ことが CUDA 最適化の鍵である。

---

## ここから行列積の最適化を始める

以降、$C = \alpha A B + \beta C$ の単精度行列積（SGEMM）を実装・最適化していく。行列サイズは $M \times K$ と $K \times N$ から $M \times N$ の結果を得る。

### 性能のロードマップ

最適化の各段階で得られる性能を先に示す（NVIDIA A6000, 4096×4096 行列）:

| Level | カーネル | GFLOPs/s | 対 cuBLAS 比 |
|-------|---------|----------|-------------|
| 1 | ナイーブ | 309 | 1.3% |
| 2 | メモリコアレッシング | 1,987 | 8.5% |
| 3 | 共有メモリキャッシュ | 2,980 | 12.8% |
| 4 | 1D Blocktiling | 8,475 | 36.5% |
| 5 | 2D Blocktiling | 15,972 | 68.7% |
| 6 | ベクトル化アクセス | 18,237 | 78.4% |
| 7 | Warptiling | 21,779 | 93.7% |

ナイーブ実装からたった7ステップで **70倍** の高速化を達成する。

---

## Level 1: ナイーブな行列積カーネル

### 発想

最もシンプルな実装: 出力行列 $C$ の各要素 $(i, j)$ を1つのスレッドが計算する。

```
C[i][j] = Σ_k  A[i][k] * B[k][j]     (k = 0, 1, ..., K-1)
```

### コード

```cuda
__global__ void sgemm_naive(int M, int N, int K,
                             float alpha, const float *A, const float *B,
                             float beta, float *C) {
    // このスレッドが担当する出力要素の座標
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // 行
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // 列

    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// 起動
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
dim3 blockDim(32, 32);  // 32×32 = 1024スレッド/ブロック
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

> **補足: `dim3` とは？**
> `dim3` は CUDA の組み込み型で、3次元のインデックスを指定する。`dim3(32, 32)` とすると 32×32 の2次元ブロックが作られ、各スレッドは `threadIdx.x` と `threadIdx.y` で自分の位置を知る。

> **補足: `CEIL_DIV` マクロ**
> `#define CEIL_DIV(a, b) ((a + b - 1) / b)` — 切り上げ除算。行列サイズがブロックサイズで割り切れない場合でも全要素をカバーするために使う。

### なぜ遅いのか？: **309 GFLOPs/s（cuBLAS の 1.3%）**

この実装は動作するが、GPU の理論性能の 1% しか引き出せていない。主な原因は **メモリアクセスパターンの非効率性** である。次の Level で詳しく見ていく。

---

## Level 2: グローバルメモリコアレッシング

### 問題の理解: なぜメモリアクセスパターンが重要なのか

GPU のグローバルメモリ（HBM）は、**同一ワープ内の連続するスレッドが連続するメモリアドレスにアクセスすると、複数のロードを1回のトランザクションにまとめる（コアレッシング）** という最適化を行う。

```
良いパターン（コアレッシング発生）:
Thread 0 → addr 0x1000    ┐
Thread 1 → addr 0x1004    ├→ 1回の128-bitトランザクション
Thread 2 → addr 0x1008    │
Thread 3 → addr 0x100C    ┘

悪いパターン（コアレッシングなし）:
Thread 0 → addr 0x1000    → 個別トランザクション
Thread 1 → addr 0x2000    → 個別トランザクション
Thread 2 → addr 0x3000    → 個別トランザクション
Thread 3 → addr 0x4000    → 個別トランザクション
```

Level 1 のナイーブ実装では、同一ワープ内のスレッドが $B$ 行列の **異なる行** にアクセスしてしまい、コアレッシングが効かない。結果として理論帯域幅 768 GB/s に対してわずか 15 GB/s しか達成できていない。

### 修正

ブロックを2次元ではなく1次元にし、スレッドのインデックス計算を変更して、同一ワープのスレッドが $B$ 行列の同一行の連続する列にアクセスするようにする。

```cuda
__global__ void sgemm_coalesced(int M, int N, int K,
                                 float alpha, const float *A, const float *B,
                                 float beta, float *C) {
    // 1Dブロック内のスレッドIDから2D座標を計算
    // 重要: threadIdx.x % BLOCKSIZE で列方向を割り当て
    //        → 同一ワープ内の連続スレッドが連続するメモリにアクセス
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);  // 行
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  // 列

    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// 起動: 1Dブロック
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
dim3 blockDim(32 * 32);  // 1024スレッドの1Dブロック
```

> **補足: Row-major レイアウトとコアレッシング**
> C/C++ の2次元配列は row-major（行優先）で格納される。つまり `B[i][0], B[i][1], B[i][2], ...` はメモリ上で連続する。同一ワープのスレッドがこれらの連続要素にアクセスすれば、GPU は複数の32-bitロードを1回の128-bitトランザクションにまとめられる。

### 結果: **1,987 GFLOPs/s（6.4倍の高速化）**

たった1行のインデックス計算の変更で、メモリ帯域幅の利用率が 15 GB/s → 110 GB/s に改善した。しかしまだ理論帯域幅の 14% しか使えていない。次のステップでは、高速な共有メモリを活用する。

---

## Level 3: 共有メモリ（SMEM）によるキャッシュブロッキング

### アイデア

グローバルメモリ（HBM）からのロードは遅い（~400 cycle）が、共有メモリ（SMEM）は高速（~20 cycle）である。**行列の一部分（タイル）を SMEM にロードし、そのタイルに対する計算を繰り返し行う** ことで、GMEM アクセス回数を大幅に削減できる。

```
         K
    ┌─────────────┐
    │             │   A: M×K
  M │    [Tile]   │
    │      ↓      │
    └─────────────┘
          ×
    ┌─────────────┐
    │   [Tile]→   │   B: K×N
  K │             │
    └─────────────┘
          ↓
    ┌─────────────┐
    │  [Result]   │   C: M×N
  M │             │
    └─────────────┘

K次元に沿って BLOCKSIZE ずつタイルをスライドさせ、
各タイルを SMEM にロード → 計算 → 次のタイルへ
```

### コード

```cuda
__global__ void sgemm_smem(int M, int N, int K,
                            float alpha, const float *A, const float *B,
                            float beta, float *C) {
    // 共有メモリの宣言（ブロック内の全スレッドで共有）
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // このブロックが担当する出力タイルの位置
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // ブロック内でのスレッド位置
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    // ポインタをブロックの開始位置に進める
    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;

    // K次元に沿ってタイルをスライド
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Phase 1: GMEM → SMEM にタイルをロード（全スレッドが協力）
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // 全スレッドのロード完了を待つ
        __syncthreads();

        // ポインタを次のタイルに進める
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // Phase 2: SMEM 上のデータで計算（高速！）
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx]
                 * Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // 次のタイルロード前に、全スレッドの計算完了を待つ
        __syncthreads();
    }

    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}
```

> **補足: `__shared__` メモリとは？**
> `__shared__` で宣言した変数は **共有メモリ (SMEM)** に配置される。SMEM はブロック内の全スレッドからアクセスでき、グローバルメモリの約15倍高速である。ただし容量は SM あたり 48〜164 KB と小さい。

> **補足: `__syncthreads()` の重要性**
> ブロック内の全スレッドを同期するバリア命令。2箇所で使用している理由:
> 1. **ロード後**: 全スレッドが SMEM へのロードを完了してから計算を開始するため
> 2. **計算後**: 全スレッドが SMEM のデータを使い終わってから次のタイルで上書きするため
>
> これを忘れると **レースコンディション** が発生し、不正な計算結果になる。

### GMEM アクセス削減の定量分析

- **Level 1**: 各出力要素の計算で $A$, $B$ から合計 $2K$ 回の GMEM ロード
- **Level 3**: $K / \text{BLOCKSIZE}$ 回のタイル反復 × 2回のロード/スレッド = $2K / \text{BLOCKSIZE}$ 回

BLOCKSIZE=32 の場合、GMEM アクセスが **1/32** に削減される。

### 結果: **2,980 GFLOPs/s（1.5倍の改善）**

コアレッシングと SMEM キャッシュを組み合わせたが、まだ cuBLAS の 12.8% に留まっている。問題は **演算強度（Arithmetic Intensity）** が低いこと — つまり、ロードしたデータに対する計算量が少なすぎる。

---

## Level 4: 1D Blocktiling — スレッドあたりの仕事量を増やす

### ボトルネック分析

Level 3 では、各スレッドが出力行列の **1要素** だけを計算する。これはスレッド数に対して計算量が少なく、メモリアクセスのレイテンシを隠蔽できない。

**解決策: 各スレッドが複数の出力要素を計算する。**

```
Level 3: 1スレッド → 1要素
[t0][t1][t2][t3]...

Level 4: 1スレッド → TM要素（縦方向に TM=8 個）
[t0] ← thread 0 が8要素を計算
[t0]
[t0]
[t0]
[t0]
[t0]
[t0]
[t0]
[t1] ← thread 1 が次の8要素
...
```

### コード

```cuda
// BM, BN: ブロックタイルのサイズ（SMEM上）
// BK: K方向のタイルサイズ
// TM: 各スレッドが縦方向に計算する要素数
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K,
                                      float alpha, const float *A, const float *B,
                                      float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // 各スレッドが担当する出力の列と、TM個分の行の開始位置
    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // SMEMロード用のインデックス
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;

    // 各スレッドが TM 個の結果を保持
    float threadResults[TM] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // GMEM → SMEM ロード
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        // 計算: ドット積の外側ループを先にする（レジスタキャッシュのため）
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // B の値をレジスタにキャッシュ（TM回のイテレーションで再利用）
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx]
            + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}
```

> **補足: ループ順序の重要性**
> 内側ループを `resIdx`（TM方向）、外側ループを `dotIdx`（K方向）にしている。これにより `Bs[dotIdx * BN + threadCol]` の値をレジスタに1回ロードするだけで TM 回再利用できる。ループ順序を逆にすると SMEM アクセスが増加し性能が落ちる。

> **補足: テンプレートパラメータを使う理由**
> `BM`, `BN`, `BK`, `TM` をテンプレートパラメータにすることで、コンパイラがこれらを定数として扱い、ループのアンロールや定数畳み込みなどの最適化を適用できる。

### 結果: **8,475 GFLOPs/s（2.8倍の改善、cuBLAS の 36.5%）**

スレッドあたりの計算量を増やすことで、メモリレイテンシを計算でマスク（隠蔽）できるようになった。GMEM アクセスに対する計算量の比（**演算強度**）が上がり、**compute-bound** に近づいている。

---

## Level 5: 2D Blocktiling — 外積による計算

### アイデア

Level 4 では縦方向 (TM) にのみスレッドの仕事量を増やしたが、**横方向 (TN) にも広げる** ことで、さらに演算強度を高める。各スレッドが $TM \times TN$ の出力タイルを計算する。

```
1スレッドの担当領域:

Level 4:  TM×1 = 8要素     Level 5:  TM×TN = 8×8 = 64要素
[*]                         [* * * * * * * *]
[*]                         [* * * * * * * *]
[*]                         [* * * * * * * *]
[*]                         [* * * * * * * *]
[*]                         [* * * * * * * *]
[*]                         [* * * * * * * *]
[*]                         [* * * * * * * *]
[*]                         [* * * * * * * *]
```

### 外積 (Outer Product) による計算

各 dotIdx に対して、$A$ の列ベクトル（TM要素）と $B$ の行ベクトル（TN要素）の外積を累積する。

```
        TN
      [b0 b1 b2 ... b7]     ← B から TN 要素をレジスタにロード
  TM
  [a0]  [a0*b0  a0*b1  ...  a0*b7]
  [a1]  [a1*b0  a1*b1  ...  a1*b7]
  [a2]  [a2*b0  a2*b1  ...  a2*b7]
  ...   [...                     ]
  [a7]  [a7*b0  a7*b1  ...  a7*b7]
   ↑
   A から TM 要素を          → TM×TN の外積 = 64回の FMA
   レジスタにロード
```

### コード

```cuda
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K,
                                      float alpha, const float *A, const float *B,
                                      float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // ブロック内のスレッド数
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // 各スレッドの2D位置（TM×TN タイル単位）
    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    // SMEMロード用（複数行をロードする必要がある場合のストライド）
    const uint strideA = numThreadsBlocktile / BK;
    const uint strideB = numThreadsBlocktile / BN;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // TM×TN の結果 + レジスタキャッシュ
    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};  // A のレジスタキャッシュ
    float regN[TN] = {0.0f};  // B のレジスタキャッシュ

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // GMEM → SMEM ロード（各スレッドが複数行を担当）
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // 外積の累積
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // A から TM 要素をレジスタにロード
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            // B から TN 要素をレジスタにロード
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            // 外積を計算して累積
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // 結果を GMEM に書き戻し
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN]
                + beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}
```

> **補足: 外積が効率的な理由**
> $A$ から TM 要素、$B$ から TN 要素をロードするだけで、$TM \times TN$ 回の FMA（積和演算）が実行できる。つまり $(TM + TN)$ 回の SMEM ロードで $TM \times TN$ 回の計算が行える。TM=TN=8 の場合、16回のロードで64回の計算 → 演算強度 4倍。

### GMEM アクセスの削減量

- Level 3: 出力要素あたり $K / \text{BLOCKSIZE}$ 回の GMEM ロード
- Level 5: 出力要素あたり $K / (TM \times TN \times \text{BLOCKSIZE} / 2)$ 回

TM=TN=8, BLOCKSIZE=32 で **$K/64$** 回まで削減（Level 3 の 1/4）。

### 結果: **15,972 GFLOPs/s（1.9倍の改善、cuBLAS の 68.7%）**

2D タイリングにより演算強度が大幅に向上し、ようやく cuBLAS の 7 割に到達した。

---

## Level 6: ベクトル化メモリアクセス

### 128-bit ベクトルロード

GPU は 32-bit の `float` を1つずつロードする代わりに、`float4`（128-bit = 4つの `float`）を一度にロードできる。これにより命令数が削減され、メモリ帯域幅の利用効率が向上する。

### 最適化 1: GMEM からのベクトルロード

```cuda
// Before: 32-bit ロード × 4回
float a0 = A[row * K + col];
float a1 = A[row * K + col + 1];
float a2 = A[row * K + col + 2];
float a3 = A[row * K + col + 3];

// After: 128-bit ロード × 1回
float4 tmp = reinterpret_cast<float4 *>(&A[row * K + col * 4])[0];
// tmp.x, tmp.y, tmp.z, tmp.w にそれぞれ値が入る
```

### 最適化 2: SMEM レイアウトの転置

$A$ 行列を SMEM に格納する際に **転置** して格納する。これにより、計算フェーズでの SMEM 読み出しが連続アドレスになり、128-bit ベクトルロード (`LDS.128`) が使えるようになる。

```cuda
// A を転置して SMEM に格納
float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;  // 転置!
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

// B はそのまま float4 でコピー
reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
```

> **補足: `reinterpret_cast<float4 *>` はなぜ必要か？**
> CUDA コンパイラは、ユーザが管理するポインタのアライメント（128-bit 境界に揃っているか）をコンパイル時に検証できない。明示的に `float4 *` にキャストすることで、「このアドレスは128-bitアラインされている」とコンパイラに伝え、ベクトルロード命令の生成を可能にする。

### 結果: **18,237 GFLOPs/s（cuBLAS の 78.4%）**

---

## Level 7: Warptiling — ワープレベルの局所性

### 新しい視点: 計算の階層化

ここまで **Grid → Block → Thread** の2階層で考えてきたが、実際の GPU ハードウェアにはもう一つ重要な階層がある: **ワープ (Warp)** である。

```
計算階層:
Grid
 └── Block     (SMEM を共有)
      └── Warp      (32スレッド、SIMT実行)
           └── Thread    (レジスタ)
```

### Warptiling のアイデア

ブロックタイルをさらにワープ単位のサブタイルに分割し、**各ワープが SMEM の特定領域に集中してアクセス** するようにする。これによりレジスタキャッシュの局所性が向上し、SMEM のバンクコンフリクトも軽減される。

```
ブロックタイル (BM × BN)
┌──────────────────────────────────┐
│ Warp 0 タイル │ Warp 1 タイル    │
│  (WM × WN)   │  (WM × WN)      │
├───────────────┼──────────────────┤
│ Warp 2 タイル │ Warp 3 タイル    │
│  (WM × WN)   │  (WM × WN)      │
└──────────────────────────────────┘

各ワープタイル内で、スレッドはさらに TM×TN の
サブタイルを複数回に分けて計算 (WMITER × WNITER 回)
```

### コードの核心部分

```cuda
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // ワープ内の各スレッドが担当するレジスタへのロード
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint i = 0; i < TM; ++i) {
            regM[wSubRowIdx * TM + i] =
                As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM
                   + threadRowInWarp * TM + i];
        }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint i = 0; i < TN; ++i) {
            regN[wSubColIdx * TN + i] =
                Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN
                   + threadColInWarp * TN + i];
        }
    }

    // ワープサブタイル全体の外積計算
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN)
                                  + (wSubColIdx * TN) + resIdxN] +=
                        regM[wSubRowIdx * TM + resIdxM] *
                        regN[wSubColIdx * TN + resIdxN];
                }
            }
        }
    }
}
```

> **補足: なぜワープを意識する必要があるのか？**
> GPU は32スレッドのワープ単位で命令を発行する（SIMT）。同一ワープ内のスレッドが SMEM の近接領域にアクセスすれば、L1 キャッシュのヒット率が上がり、バンクコンフリクトも減る。逆に、ワープ内のスレッドがバラバラの SMEM 領域にアクセスすると性能が劣化する。

> **補足: ILP (Instruction-Level Parallelism)**
> ループを明示的にネストすることで、コンパイラが独立した FMA 命令を並列発行しやすくなる。これにより GPU の演算パイプラインの利用率が向上し、レジスタ→SMEM のレイテンシもマスクできる。

### 結果: **21,779 GFLOPs/s（cuBLAS の 93.7%）**

ナイーブ実装の **70倍** 、cuBLAS の **93.7%** の性能を達成した。

---

## Level 8: Hopper 世代の Tensor Core（WGMMA）

### 次のフロンティア

ここまでの最適化は **CUDA Core**（FP32 の FMA 演算器）を使ったものである。NVIDIA Hopper (H100) 以降では、**Tensor Core** という専用行列演算ユニットが使える。Tensor Core は CUDA Core に比べて数倍〜数十倍のスループットを持つ。

### WGMMA (Warp Group Matrix Multiply-Accumulate)

Hopper の Tensor Core を使うための命令が **WGMMA** である。従来の `wmma` 命令（Volta/Turing/Ampere）と異なり、以下の特徴がある:

- **128スレッド（4ワープ）** が協調して1つの行列積を実行
- **非同期実行**: 計算と次のデータロードをオーバーラップ可能
- **SMEM から直接オペランドを読む**: レジスタへの明示的なロードが不要（ディスクリプタ方式）

```
WGMMA の動作:
                SMEM
                 ↓
  [Warp 0]  ─┐
  [Warp 1]  ─┼→ Tensor Core → C (レジスタ)
  [Warp 2]  ─┤    64×64×16
  [Warp 3]  ─┘    の行列積
                 ↑
                SMEM
```

### CUTLASS による抽象化

WGMMA を直接 PTX で書くのは非常に煩雑なため、NVIDIA の [CUTLASS](https://github.com/NVIDIA/cutlass) ライブラリが提供する `cute::gemm` API を使う。

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

// TiledMMA の定義
// SM90: Hopper, 64x64x16: タイルサイズ, F16F16F16: データ型, SS: A,B共にSMEMから
using MMA = decltype(make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{}
));
```

> **補足: `SM90_64x64x16_F16F16F16_SS` の読み方**
> - `SM90`: Hopper アーキテクチャ (compute capability 9.0)
> - `64x64x16`: 1回の WGMMA 命令で計算する $M \times N \times K$ のサイズ
> - `F16F16F16`: 入力 A が FP16、入力 B が FP16、アキュムレータが FP16
> - `_SS`: A, B 共に Shared Memory から読み込む（`_RS` なら A はレジスタから）

### SMEM レイアウトの特殊性

Hopper の Tensor Core は SMEM に対して **特殊なスウィズルレイアウト** を要求する。単純な row-major や column-major では動作しない。

```cpp
// CUTLASS が提供するスウィズルレイアウト
using SmemLayoutA = decltype(
    tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},  // 128-byte スウィズル
        Shape<Int<BM>, Int<BK>>{}
    )
);
```

> **補足: スウィズル (Swizzle) とは？**
> SMEM のバンクコンフリクトを回避するため、アドレスのビットを並べ替えるテクニック。Hopper の Tensor Core は特定のスウィズルパターンを前提にハードウェアが設計されているため、正しいレイアウトを使わないと不正な結果になる。

### カーネルの基本構造

```cpp
__global__ void gemm_wgmma(/* ... */) {
    // === Prologue ===
    // SMEM の確保とレイアウト設定
    extern __shared__ char smem[];
    auto sA = make_tensor(make_smem_ptr(smem), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem + size(sA)), SmemLayoutB{});

    // TiledMMA の構築
    MMA tiled_mma;
    auto tCrC = tiled_mma.partition_fragment_C(sC);  // アキュムレータ
    clear(tCrC);

    // === Mainloop ===
    for (int k_tile = 0; k_tile < K / BK; ++k_tile) {
        // GMEM → SMEM ロード（TMA を使った非同期コピーが理想的）
        // ... load tiles ...

        // WGMMA 実行
        warpgroup_arrive();
        gemm(tiled_mma, tCrA(_, _, k_tile), tCrB(_, _, k_tile), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    // === Epilogue ===
    // レジスタ → GMEM に結果を書き戻し
    // ... store results ...
}
```

> **補足: `warpgroup_arrive/commit_batch/wait` とは？**
> WGMMA は **非同期命令** である。128スレッド（4ワープ）が協調して動作するため、以下の同期が必要:
> - `warpgroup_arrive()`: 全ワープが到達したことを通知
> - `warpgroup_commit_batch()`: WGMMA 命令をバッチとしてコミット
> - `warpgroup_wait<0>()`: コミットした全バッチの完了を待機

### TMA (Tensor Memory Accelerator)

Hopper にはもう一つの強力な機能がある: **TMA** はグローバルメモリから共有メモリへのデータ転送を、スレッドの介在なしにハードウェアで行う。

```
従来 (Ampere以前):
  スレッドが GMEM → レジスタ → SMEM にコピー
  （スレッドの命令スロットを消費）

Hopper (TMA):
  TMA ユニットが GMEM → SMEM を直接転送
  （スレッドは計算に専念できる）
```

これにより、**データ転送と計算の完全なオーバーラップ** が可能になり、理論性能に近い実効性能を達成できる。

---

## まとめ: 最適化の全体像

本記事で辿った最適化の旅を振り返る:

| Level | 技法 | 核心アイデア | 対 cuBLAS |
|-------|------|------------|-----------|
| 1 | ナイーブ | 1スレッド1要素 | 1.3% |
| 2 | コアレッシング | 連続スレッド→連続メモリ | 8.5% |
| 3 | SMEM キャッシュ | 高速メモリにデータを置く | 12.8% |
| 4 | 1D Blocktiling | スレッドの仕事量を増やす | 36.5% |
| 5 | 2D Blocktiling | 外積で演算強度を上げる | 68.7% |
| 6 | ベクトル化 | 128-bit ロードで帯域効率UP | 78.4% |
| 7 | Warptiling | ワープ内の局所性を活用 | 93.7% |
| 8 | WGMMA | Tensor Core で桁違いの性能 | — |

### 学んだ原則

1. **メモリアクセスパターンが全て**: GPU の性能はメモリ帯域幅で律速されることが多い。コアレッシング、SMEM キャッシュ、ベクトルロードはすべてメモリ効率の改善
2. **演算強度を上げる**: ロードしたデータに対してできるだけ多くの計算を行う。1D → 2D Blocktiling はその典型
3. **ハードウェア階層を意識する**: レジスタ > SMEM > L2 > GMEM の速度差を理解し、データを適切な階層に置く
4. **ワープを理解する**: 32スレッドのワープが GPU の真の実行単位。コアレッシング、バンクコンフリクト、SIMT 実行はすべてワープ単位

### 次のステップ

本記事で学んだ知識を活かして、以下に挑戦してみてほしい:

- [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA): 本記事の参考元。全カーネルのソースコードとベンチマーク
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass): プロダクション品質の GEMM ライブラリ。内部実装の理解に最適
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): NVIDIA 公式リファレンス
- [Colfax Research CUTLASS Tutorial](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/): Hopper + WGMMA の実践的チュートリアル

---

## 参考文献

- Simon Boehm, "[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)"
- Mark Harris, "[An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)"
- Colfax Research, "[CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)"

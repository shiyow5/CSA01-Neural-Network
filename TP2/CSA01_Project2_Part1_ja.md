# CSA01 - チームプロジェクト2 パート1: 多層パーセプトロンと誤差逆伝播法

**科目:** ニューラルネットワーク (CSA01)  
**チームメンバー:**
- 佐藤 丞 (m5301059)
- 宇佐美 雄貴 (m5301073)
- 関根 健人 (m5301060)
- 相澤 祐真 (m5301001)
- 渡部 千歳 (m5301074)

---

## a) 解いた問題

今回は誤差逆伝播法（BP）を用いた3層ニューラルネットワークを実装し、**4ビットパリティチェック問題**で動作を検証した。

パリティ関数は、4つの入力ビットのうち1の個数が偶数なら1、奇数なら0を出力するものだ。線形分離不可能な問題の代表例で、単層ネットワークでは解けないため、多層ネットワークのベンチマークとしてよく使われる。

ネットワークの構成は次の通り:
- **入力層:** 5ユニット（4ビット＋バイアス用ダミー入力、値は-1固定）
- **隠れ層:** ニューロン数を可変（4, 6, 8, 10で比較）
- **出力層:** 1ニューロン、出力範囲は[0, 1]

訓練サンプルは2^4 = 16通り:

| パターン | x1 | x2 | x3 | x4 | バイアス | 1の個数 | 期待出力 |
|:-------:|:--:|:--:|:--:|:--:|:------:|:------:|:-------:|
| 0       | 0  | 0  | 0  | 0  | -1     | 0（偶数）| 1       |
| 1       | 0  | 0  | 0  | 1  | -1     | 1（奇数）| 0       |
| 2       | 0  | 0  | 1  | 0  | -1     | 1（奇数）| 0       |
| 3       | 0  | 0  | 1  | 1  | -1     | 2（偶数）| 1       |
| 4       | 0  | 1  | 0  | 0  | -1     | 1（奇数）| 0       |
| 5       | 0  | 1  | 0  | 1  | -1     | 2（偶数）| 1       |
| 6       | 0  | 1  | 1  | 0  | -1     | 2（偶数）| 1       |
| 7       | 0  | 1  | 1  | 1  | -1     | 3（奇数）| 0       |
| 8       | 1  | 0  | 0  | 0  | -1     | 1（奇数）| 0       |
| 9       | 1  | 0  | 0  | 1  | -1     | 2（偶数）| 1       |
| 10      | 1  | 0  | 1  | 0  | -1     | 2（偶数）| 1       |
| 11      | 1  | 0  | 1  | 1  | -1     | 3（奇数）| 0       |
| 12      | 1  | 1  | 0  | 0  | -1     | 2（偶数）| 1       |
| 13      | 1  | 1  | 0  | 1  | -1     | 3（奇数）| 0       |
| 14      | 1  | 1  | 1  | 0  | -1     | 3（奇数）| 0       |
| 15      | 1  | 1  | 1  | 1  | -1     | 4（偶数）| 1       |

### ソースコード: 誤差逆伝播法 (`multilayer_perceptron.c`)

```c
/************************************************************************************/
/* C-program for BP algorithm                                                       */
/* The nerual network to be designed is supposed to have                            */
/* three layers:                                                                    */
/*  1) Input layer : I inputs                                                       */
/*  2) Hidden layer: J neurons                                                      */
/*  3) Output layer: K neurons                                                      */
/* The last input is always -1, and the output of the last                          */
/* hidden neuron is also -1.                                                        */
/*                                                                                  */
/* This program is produced by Qiangfu Zhao and extended by m5301059 SATO Sho.      */
/* You are free to use it for educational purpose                                   */
/************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I 5
#define J 5
#define K 1
#define n_sample 16
#define eta 0.5
#define lambda 1.0
#define desired_error 0.001
#define sigmoid(x) (1.0 / (1.0 + exp(-lambda * x)))
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

double x[n_sample][I] = {
    {0, 0, 0, 0, -1},
    {0, 0, 0, 1, -1},
    {0, 0, 1, 0, -1},
    {0, 0, 1, 1, -1},
    {0, 1, 0, 0, -1},
    {0, 1, 0, 1, -1},
    {0, 1, 1, 0, -1},
    {0, 1, 1, 1, -1},
    {1, 0, 0, 0, -1},
    {1, 0, 0, 1, -1},
    {1, 0, 1, 0, -1},
    {1, 0, 1, 1, -1},
    {1, 1, 0, 0, -1},
    {1, 1, 0, 1, -1},
    {1, 1, 1, 0, -1},
    {1, 1, 1, 1, -1}};
double d[n_sample][K] = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1};
double v[J][I], w[K][J];
double y[J];
double o[K];

void Initialization(void);
void FindHidden(int p);
void FindOutput(void);
void PrintResult(void);

int main()
{
    int i, j, k, p, q = 0;
    double Error = DBL_MAX;
    double delta_o[K];
    double delta_y[J];

    Initialization();
    while (Error > desired_error)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            FindHidden(p);
            FindOutput();

            for (k = 0; k < K; k++)
            {
                Error += 0.5 * pow(d[p][k] - o[k], 2.0);
                delta_o[k] = (d[p][k] - o[k]) * (1 - o[k]) * o[k];
            }

            for (j = 0; j < J; j++)
            {
                delta_y[j] = 0;
                for (k = 0; k < K; k++)
                    delta_y[j] += delta_o[k] * w[k][j];
                delta_y[j] = (1 - y[j]) * y[j] * delta_y[j];
            }

            for (k = 0; k < K; k++)
                for (j = 0; j < J; j++)
                    w[k][j] += eta * delta_o[k] * y[j];

            for (j = 0; j < J; j++)
                for (i = 0; i < I; i++)
                    v[j][i] += eta * delta_y[j] * x[p][i];
        }
        printf("Error in the %d-th learning cycle = %f\n", q, Error);
    }

    PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
    int i, j, k;

    randomize();
    for (j = 0; j < J; j++)
        for (i = 0; i < I; i++)
            v[j][i] = frand() - 0.5;

    for (k = 0; k < K; k++)
        for (j = 0; j < J; j++)
            w[k][j] = frand() - 0.5;
}

/*************************************************************/
/* Find the output of the hidden neurons                     */
/*************************************************************/
void FindHidden(int p)
{
    int i, j;
    double temp;

    for (j = 0; j < J - 1; j++)
    {
        temp = 0;
        for (i = 0; i < I; i++)
            temp += v[j][i] * x[p][i];
        y[j] = sigmoid(temp);
    }
    y[J - 1] = -1;
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(void)
{
    int j, k;
    double temp;

    for (k = 0; k < K; k++)
    {
        temp = 0;
        for (j = 0; j < J; j++)
            temp += w[k][j] * y[j];
        o[k] = sigmoid(temp);
    }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void)
{
    int i, j, k, p;

    printf("\n\n");
    printf("The connection weights in the output layer:\n");
    for (k = 0; k < K; k++)
    {
        for (j = 0; j < J; j++)
            printf("%5f ", w[k][j]);
        printf("\n");
    }

    printf("\n\n");
    printf("The connection weights in the hidden layer:\n");
    for (j = 0; j < J - 1; j++)
    {
        for (i = 0; i < I; i++)
            printf("%5f ", v[j][i]);
        printf("\n");
    }
    printf("\n\n");

    printf("Neuron output for each input pattern:\n");
  for (p = 0; p < n_sample; p++)
  {
    FindHidden(p);
    FindOutput();
    printf("(");
    for (i = 0; i < I; i++)
      printf(" %.1f,", x[p][i]);
    printf(") -> (");
    for (i = 0; i < K; i++)
      printf(" %5f,", o[i]);
    printf(")\n");
  }
  printf("\n");
}
```

### ソースコード: 実験スクリプト (`multilayer_perceptron_exp.c`)

隠れ層のニューロン数を比較するために、コンパイル時に `-DJ_VAL=` で隠れ層サイズ、コマンドライン引数で乱数シードを指定できるバージョンを作成した。これにより、異なる初期化条件で複数回の試行を自動化できる。

```c
/************************************************************************************/
/* C-program for BP algorithm (experiment version)                                 */
/* Compile with: gcc multilayer_perceptron_exp.c -o bp_exp.out -lm -DJ_VAL=5       */
/* Usage: ./bp_exp.out [seed]                                                      */
/*                                                                                 */
/* J_VAL includes 1 bias neuron, so hidden neurons = J_VAL - 1.                    */
/* This program is produced by Qiangfu Zhao and extended by m5301059 SATO Sho.     */
/************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define I 5
#ifndef J_VAL
#define J_VAL 5
#endif
#define J J_VAL
#define K 1
#define n_sample 16
#define eta 0.5
#define lambda 1.0
#define desired_error 0.001
#ifndef MAX_EPOCH
#define MAX_EPOCH 50000
#endif
#define sigmoid(x) (1.0 / (1.0 + exp(-lambda * (x))))
#define frand() (rand() % 10000 / 10001.0)

double x[n_sample][I] = {
    {0, 0, 0, 0, -1},
    {0, 0, 0, 1, -1},
    {0, 0, 1, 0, -1},
    {0, 0, 1, 1, -1},
    {0, 1, 0, 0, -1},
    {0, 1, 0, 1, -1},
    {0, 1, 1, 0, -1},
    {0, 1, 1, 1, -1},
    {1, 0, 0, 0, -1},
    {1, 0, 0, 1, -1},
    {1, 0, 1, 0, -1},
    {1, 0, 1, 1, -1},
    {1, 1, 0, 0, -1},
    {1, 1, 0, 1, -1},
    {1, 1, 1, 0, -1},
    {1, 1, 1, 1, -1}};
double d[n_sample] = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1};
double v[J][I], w[K][J];
double y[J];
double o[K];

void Initialization(unsigned int seed)
{
    int i, j, k;
    srand(seed);
    for (j = 0; j < J; j++)
        for (i = 0; i < I; i++)
            v[j][i] = frand() - 0.5;
    for (k = 0; k < K; k++)
        for (j = 0; j < J; j++)
            w[k][j] = frand() - 0.5;
}

void FindHidden(int p)
{
    int i, j;
    double temp;
    for (j = 0; j < J - 1; j++)
    {
        temp = 0;
        for (i = 0; i < I; i++)
            temp += v[j][i] * x[p][i];
        y[j] = sigmoid(temp);
    }
    y[J - 1] = -1;
}

void FindOutput(void)
{
    int j, k;
    double temp;
    for (k = 0; k < K; k++)
    {
        temp = 0;
        for (j = 0; j < J; j++)
            temp += w[k][j] * y[j];
        o[k] = sigmoid(temp);
    }
}

int main(int argc, char *argv[])
{
    int i, j, k, p, q = 0;
    double Error = DBL_MAX;
    double delta_o[K];
    double delta_y[J];
    unsigned int seed;

    if (argc > 1)
        seed = (unsigned int)atoi(argv[1]);
    else
        seed = 42;

    Initialization(seed);

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            FindHidden(p);
            FindOutput();

            for (k = 0; k < K; k++)
            {
                Error += 0.5 * pow(d[p] - o[k], 2.0);
                delta_o[k] = (d[p] - o[k]) * (1 - o[k]) * o[k];
            }

            for (j = 0; j < J - 1; j++)
            {
                delta_y[j] = 0;
                for (k = 0; k < K; k++)
                    delta_y[j] += delta_o[k] * w[k][j];
                delta_y[j] = (1 - y[j]) * y[j] * delta_y[j];
            }

            for (k = 0; k < K; k++)
                for (j = 0; j < J; j++)
                    w[k][j] += eta * delta_o[k] * y[j];

            for (j = 0; j < J - 1; j++)
                for (i = 0; i < I; i++)
                    v[j][i] += eta * delta_y[j] * x[p][i];
        }
    }

    /* Count correct classifications */
    int correct = 0;
    for (p = 0; p < n_sample; p++)
    {
        FindHidden(p);
        FindOutput();
        int predicted = (o[0] >= 0.5) ? 1 : 0;
        int expected = (int)d[p];
        if (predicted == expected)
            correct++;
    }

    printf("EPOCHS: %d\n", q);
    printf("FINAL_ERROR: %f\n", Error);
    printf("CORRECT: %d\n", correct);
    printf("STATUS: %s\n", (Error <= desired_error) ? "CONVERGED" : "NOT_CONVERGED");

    return 0;
}
```

## b) 使用した手法

### ネットワーク構造

標準的な3層フィードフォワードネットワークを使用した。
- **入力層:** 5ユニット。最初の4つが入力ビット、5つ目はバイアス用のダミー入力（値は-1固定）。
- **隠れ層:** J個のニューロン。最後の1つはバイアス用に-1に固定されるため、実質的な隠れニューロン数はJ - 1個になる。
- **出力層:** 1ニューロン、シグモイド活性化関数で出力は[0, 1]の範囲。

### 活性化関数

隠れ層・出力層ともに、標準的なロジスティックシグモイドを使用した:

```
f(x) = 1 / (1 + exp(-lambda * x)),  lambda = 1.0
```

出力範囲が[0, 1]なので、教師信号も1（偶数パリティ）と0（奇数パリティ）で設定している。

### 誤差逆伝播法のアルゴリズム

各訓練パターンについて、順伝播で隠れ層→出力層の出力を計算し、逆伝播で誤差信号を求めて重みを更新する。

**出力層の誤差信号:**
```
delta_o = (d - o) * o * (1 - o)
```

ここで `o * (1 - o)` はシグモイド関数の導関数。

**隠れ層の誤差信号:**
```
delta_y[j] = y[j] * (1 - y[j]) * SUM_k(delta_o[k] * w[k][j])
```

**重みの更新:**
```
w[k][j] += eta * delta_o[k] * y[j]     （出力層）
v[j][i] += eta * delta_y[j] * x[p][i]  （隠れ層）
```

### 学習パラメータ

- 学習率 (eta): 0.5
- 誤差閾値: 0.001
- 誤差指標: E = 0.5 * SUM(d - o)^2（16パターン全体）
- 重み初期値: [-0.5, 0.5]の乱数
- 最大エポック（実験版）: 50,000

### 実験設定

隠れ層の大きさによる性能の違いを調べるため、4, 6, 8, 10個の隠れニューロンそれぞれについて、異なるランダムシードで10試行ずつ実行した。各試行について50,000エポック以内に収束したかどうかと、収束までのエポック数を記録した。

## c) シミュレーション結果の考察

### 実験結果

`run_experiment.sh`を使い、隠れニューロン数4, 6, 8, 10（バイアス込みでJ = 5, 7, 9, 11）の4条件について、それぞれ10試行の実験を行った。

| 隠れニューロン数 | 収束率 | 平均エポック | 最小エポック | 最大エポック |
|:---------------:|:------:|:----------:|:----------:|:----------:|
| 4               | 6/10 (60%)  | 31,444 | 28,485 | 36,253 |
| 6               | 10/10 (100%) | 27,907 | 19,333 | 41,905 |
| 8               | 10/10 (100%) | 25,176 | 18,938 | 31,567 |
| 10              | 9/10 (90%)  | 28,177 | 23,385 | 38,744 |

収束しなかったケースでは、最終誤差が0.405付近に留まり、16パターン中1パターンを誤分類した状態だった。

### 考察

まず目につくのは、隠れ4個だと収束率が60%しかないことだ。10回中4回が50,000エポック以内に収束しなかった。4ビットパリティ関数は決定領域が複雑で、隠れ4個だとネットワークの表現力がぎりぎりなので、初期重みによっては局所最小解にはまってしまうのだろうと思う。

6個と8個に増やすと収束率が100%になった。ニューロンが増えるとネットワークの表現力に余裕が出るため、初期値への依存が弱くなる。平均エポック数も8個のとき~25,000で最も少なく、隠れ層が大きいほど勾配が流れやすくなる分、学習も少し速くなっている。

面白いのは、10個に増やしたとき収束率が90%に下がったことだ。最初は意外だったが、パラメータが増えると誤差曲面も複雑になり、局所解の数も増える。表現力としては十分でも、特定の初期値の組み合わせで悪い方向に引き込まれることがあるのだと考えている。収束した試行の平均エポック数も8個のときより増えている。

全体として、4ビットパリティ問題には隠れ6〜8個あたりが良さそうだ。表現力が十分に確保でき、かつパラメータが多すぎない。

もう一つ気になったのは、どの条件でも数万エポックが必要なことだ。これはパリティ関数の性質による部分が大きい。入力ビットのどれか1つを反転するだけで出力が変わるため、局所的な構造がほとんどなく、ネットワークは16パターンの真理値表を丸ごと覚える必要がある。

---

## d) 新しい問題: 8ビットパリティ

パリティ問題の難しさが入力ビット数に対してどうスケールするか確かめるため、8ビットパリティに拡張した。基本的な考え方は4ビットと同じだが、訓練パターンが256個（8ビットの全組み合わせ）になり、入力層は9ユニット（8ビット＋バイアス1つ）になる。隠れニューロン数を8, 16, 24, 32で試し、各5試行、最大100,000エポック、収束誤差0.01で実験した。

### ソースコード (`parity8_experiment.c`)

```c
/************************************************************************************/
/* BP algorithm for 8-bit parity check problem                                     */
/* Compile: gcc parity8_experiment.c -o parity8_exp.out -lm -DJ_VAL=9              */
/* Usage: ./parity8_exp.out [seed]                                                 */
/*                                                                                 */
/* 8 inputs + 1 bias = 9 input units, 256 training patterns.                       */
/* J_VAL includes 1 bias neuron, so hidden neurons = J_VAL - 1.                    */
/* This program is produced by m5301059 SATO Sho.                                  */
/************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define I 9         /* 8 bits + 1 bias */
#define n_sample 256
#ifndef J_VAL
#define J_VAL 9     /* default: 8 hidden + 1 bias */
#endif
#define J J_VAL
#define K 1
#define eta 0.5
#define lambda 1.0
#define desired_error 0.01
#ifndef MAX_EPOCH
#define MAX_EPOCH 100000
#endif
#define sigmoid(x) (1.0 / (1.0 + exp(-lambda * (x))))
#define frand() (rand() % 10000 / 10001.0)

double x[n_sample][I];
double d[n_sample];
double v[J][I], w[K][J];
double y[J];
double o[K];

int popcount(int n)
{
    int count = 0;
    while (n) { count += n & 1; n >>= 1; }
    return count;
}

void GenerateData(void)
{
    int p, i;
    for (p = 0; p < n_sample; p++)
    {
        for (i = 0; i < 8; i++)
            x[p][i] = (p >> (7 - i)) & 1;
        x[p][8] = -1; /* bias */
        d[p] = (popcount(p) % 2 == 0) ? 1.0 : 0.0;
    }
}

void Initialization(unsigned int seed)
{
    int i, j, k;
    srand(seed);
    for (j = 0; j < J; j++)
        for (i = 0; i < I; i++)
            v[j][i] = frand() - 0.5;
    for (k = 0; k < K; k++)
        for (j = 0; j < J; j++)
            w[k][j] = frand() - 0.5;
}

void FindHidden(int p)
{
    int i, j;
    double temp;
    for (j = 0; j < J - 1; j++)
    {
        temp = 0;
        for (i = 0; i < I; i++)
            temp += v[j][i] * x[p][i];
        y[j] = sigmoid(temp);
    }
    y[J - 1] = -1;
}

void FindOutput(void)
{
    int j, k;
    double temp;
    for (k = 0; k < K; k++)
    {
        temp = 0;
        for (j = 0; j < J; j++)
            temp += w[k][j] * y[j];
        o[k] = sigmoid(temp);
    }
}

int main(int argc, char *argv[])
{
    int i, j, k, p, q = 0;
    double Error = DBL_MAX;
    double delta_o[K];
    double delta_y[J];
    unsigned int seed = (argc > 1) ? (unsigned int)atoi(argv[1]) : 42;

    GenerateData();
    Initialization(seed);

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            FindHidden(p);
            FindOutput();

            for (k = 0; k < K; k++)
            {
                Error += 0.5 * pow(d[p] - o[k], 2.0);
                delta_o[k] = (d[p] - o[k]) * (1 - o[k]) * o[k];
            }

            for (j = 0; j < J - 1; j++)
            {
                delta_y[j] = 0;
                for (k = 0; k < K; k++)
                    delta_y[j] += delta_o[k] * w[k][j];
                delta_y[j] = (1 - y[j]) * y[j] * delta_y[j];
            }

            for (k = 0; k < K; k++)
                for (j = 0; j < J; j++)
                    w[k][j] += eta * delta_o[k] * y[j];

            for (j = 0; j < J - 1; j++)
                for (i = 0; i < I; i++)
                    v[j][i] += eta * delta_y[j] * x[p][i];
        }
    }

    int correct = 0;
    for (p = 0; p < n_sample; p++)
    {
        FindHidden(p);
        FindOutput();
        int predicted = (o[0] >= 0.5) ? 1 : 0;
        int expected = (int)d[p];
        if (predicted == expected)
            correct++;
    }

    printf("EPOCHS: %d\n", q);
    printf("FINAL_ERROR: %f\n", Error);
    printf("CORRECT: %d\n", correct);
    printf("STATUS: %s\n", (Error <= desired_error) ? "CONVERGED" : "NOT_CONVERGED");

    return 0;
}
```

### 実験結果

| 隠れニューロン数 | 収束率 | 最良正答率 | 典型的な誤差 |
|:---------------:|:------:|:---------:|:----------:|
| 8               | 0/5 (0%)  | 227/256 (89%) | ~10.2  |
| 16              | 0/5 (0%)  | 227/256 (89%) | ~10.2  |
| 24              | 0/5 (0%)  | 248/256 (97%) | ~3.4〜10.2 |
| 32              | 0/5 (0%)  | 248/256 (97%) | ~3.3〜10.2 |

100,000エポックでもどの条件でも収束しなかった。ほとんどの試行が227/256（89%）で止まっていて、約29パターンが常に誤分類される状態だった。24個と32個の隠れニューロンでは一部の試行が248/256（97%）まで到達したが、それでも完全な収束には至らなかった。

4ビットパリティとの難易度の差は歴然だ。4ビットから8ビットにすると訓練パターンが16倍（16→256）に増えるだけでなく、パリティ関数自体がもっと入り組んだものになる。ビット数が1つ増えるたびに真理値表の符号反転が倍増するので、入力空間はどこを見ても0と1が市松模様のように混ざり合っている。ネットワークは真理値表を丸暗記するしかないが、誤差曲面には勾配がほぼゼロの広い平坦領域が多く、通常のBPではそこから抜け出せない。こうした問題に対しては、e)で試すモメンタムのようなアルゴリズムの改良が必要になる。

---

## e) 手法の改良: モメンタム項の追加

通常のBPは、現在の勾配だけで重みを更新する。誤差曲面に細い谷がある場合、勾配が谷の壁を行ったり来たりして、谷に沿った方向になかなか進めないことがある。モメンタムはこれを改善する古典的な手法だ。

考え方はシンプルで、前回の重み更新量の一定割合を今回の更新に加える。数式で書くと:

```
Delta_w(t) = eta * delta * y  +  alpha * Delta_w(t-1)
w(t+1) = w(t) + Delta_w(t)
```

ここで`alpha`はモメンタム係数（0 <= alpha < 1）。隠れ層の重み`v`にも同様に適用する。

なぜこれが効くかというと、勾配が何ステップも同じ方向を向いているとき、モメンタム項が蓄積して実効的なステップサイズが大きくなるからだ。逆に、勾配が毎回違う方向を向いている（振動している）場合は、モメンタム項同士が打ち消し合って更新が抑えられる。つまりモメンタムは一種の適応的ステップサイズ調整として機能する。一貫した方向では加速し、振動は減衰させる。

物理のアナロジーで言えば、坂を転がるボールを想像するとわかりやすい。モメンタムなしだと、傾きがなくなった瞬間にボールは止まる。モメンタムありだと、過去のステップからの運動エネルギーを持っているので、平坦な領域や小さな窪み（局所解）を越えていける。

もう少し正確に見ると、再帰を展開すると時刻tでの実効的な重み更新は:

```
Delta_w(t) = eta * [ g(t) + alpha*g(t-1) + alpha^2*g(t-2) + ... ]
```

これは過去の勾配の指数加重移動平均になっている。係数`alpha^k`が付くので古い勾配ほど影響が薄れ、半減期はおよそ`1 / (1 - alpha)`ステップ。alpha = 0.9なら直近の約10ステップ分の勾配が効いている計算になる。

また、もし勾配が一定（毎ステップ同じ `g`）だとすると、モメンタムが定常状態に達したときの実効的な学習率は `eta / (1 - alpha)` になる。alpha = 0.9のとき、これは `0.5 / 0.1 = 5.0` で、名目の学習率の10倍だ。収束が速くなるのもうなずける。

ただし、勾配は実際には一定ではない。各訓練パターンごとに異なる勾配が生じるし、重みが変われば勾配も変わる。勾配の分散が大きい（ステップごとに向きがバラバラな）場合、モメンタム項はそのノイズを増幅してしまう。結果として重みが大きく振動したり発散したりする可能性がある。

### ソースコード (`momentum_experiment.c`)

```c
/************************************************************************************/
/* BP algorithm with momentum for 4-bit parity check problem                       */
/* Compile: gcc momentum_experiment.c -o momentum_exp.out -lm                      */
/*          -DJ_VAL=5 -DALPHA_VAL=90                                               */
/* Usage: ./momentum_exp.out [seed]                                                */
/*                                                                                 */
/* ALPHA_VAL is momentum * 100 (integer), e.g. 90 means alpha=0.9                  */
/* This program is produced by m5301059 SATO Sho.                                  */
/************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define I 5
#ifndef J_VAL
#define J_VAL 5
#endif
#define J J_VAL
#define K 1
#define n_sample 16
#define eta 0.5
#define lambda 1.0
#define desired_error 0.001
#ifndef MAX_EPOCH
#define MAX_EPOCH 50000
#endif
#ifndef ALPHA_VAL
#define ALPHA_VAL 0  /* no momentum by default */
#endif
#define alpha (ALPHA_VAL / 100.0)
#define sigmoid(x) (1.0 / (1.0 + exp(-lambda * (x))))
#define frand() (rand() % 10000 / 10001.0)

double x[n_sample][I] = {
    {0, 0, 0, 0, -1}, {0, 0, 0, 1, -1}, {0, 0, 1, 0, -1}, {0, 0, 1, 1, -1},
    {0, 1, 0, 0, -1}, {0, 1, 0, 1, -1}, {0, 1, 1, 0, -1}, {0, 1, 1, 1, -1},
    {1, 0, 0, 0, -1}, {1, 0, 0, 1, -1}, {1, 0, 1, 0, -1}, {1, 0, 1, 1, -1},
    {1, 1, 0, 0, -1}, {1, 1, 0, 1, -1}, {1, 1, 1, 0, -1}, {1, 1, 1, 1, -1}};
double d[n_sample] = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1};
double v[J][I], w[K][J];
double dv[J][I], dw[K][J]; /* previous weight changes for momentum */
double y[J];
double o[K];

void Initialization(unsigned int seed)
{
    int i, j, k;
    srand(seed);
    for (j = 0; j < J; j++)
        for (i = 0; i < I; i++)
        {
            v[j][i] = frand() - 0.5;
            dv[j][i] = 0;
        }
    for (k = 0; k < K; k++)
        for (j = 0; j < J; j++)
        {
            w[k][j] = frand() - 0.5;
            dw[k][j] = 0;
        }
}

void FindHidden(int p)
{
    int i, j;
    double temp;
    for (j = 0; j < J - 1; j++)
    {
        temp = 0;
        for (i = 0; i < I; i++)
            temp += v[j][i] * x[p][i];
        y[j] = sigmoid(temp);
    }
    y[J - 1] = -1;
}

void FindOutput(void)
{
    int j, k;
    double temp;
    for (k = 0; k < K; k++)
    {
        temp = 0;
        for (j = 0; j < J; j++)
            temp += w[k][j] * y[j];
        o[k] = sigmoid(temp);
    }
}

int main(int argc, char *argv[])
{
    int i, j, k, p, q = 0;
    double Error = DBL_MAX;
    double delta_o[K];
    double delta_y[J];
    double change;
    unsigned int seed = (argc > 1) ? (unsigned int)atoi(argv[1]) : 42;

    Initialization(seed);

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            FindHidden(p);
            FindOutput();

            for (k = 0; k < K; k++)
            {
                Error += 0.5 * pow(d[p] - o[k], 2.0);
                delta_o[k] = (d[p] - o[k]) * (1 - o[k]) * o[k];
            }

            for (j = 0; j < J - 1; j++)
            {
                delta_y[j] = 0;
                for (k = 0; k < K; k++)
                    delta_y[j] += delta_o[k] * w[k][j];
                delta_y[j] = (1 - y[j]) * y[j] * delta_y[j];
            }

            /* Update output layer weights with momentum */
            for (k = 0; k < K; k++)
                for (j = 0; j < J; j++)
                {
                    change = eta * delta_o[k] * y[j] + alpha * dw[k][j];
                    w[k][j] += change;
                    dw[k][j] = change;
                }

            /* Update hidden layer weights with momentum */
            for (j = 0; j < J - 1; j++)
                for (i = 0; i < I; i++)
                {
                    change = eta * delta_y[j] * x[p][i] + alpha * dv[j][i];
                    v[j][i] += change;
                    dv[j][i] = change;
                }
        }
    }

    int correct = 0;
    for (p = 0; p < n_sample; p++)
    {
        FindHidden(p);
        FindOutput();
        int predicted = (o[0] >= 0.5) ? 1 : 0;
        int expected = (int)d[p];
        if (predicted == expected)
            correct++;
    }

    printf("EPOCHS: %d\n", q);
    printf("FINAL_ERROR: %f\n", Error);
    printf("CORRECT: %d\n", correct);
    printf("STATUS: %s\n", (Error <= desired_error) ? "CONVERGED" : "NOT_CONVERGED");

    return 0;
}
```

### 実験結果

4ビットパリティ問題に対し、隠れ4個（J = 5）でモメンタム係数alphaを5段階変えて各10試行ずつ実施した。

| モメンタム (alpha) | 収束率 | 平均エポック | 最小 | 最大 |
|:-----------------:|:------:|:----------:|:----:|:----:|
| 0.0（ベースライン）| 6/10 (60%)  | 31,444 | 28,485 | 36,253 |
| 0.3               | 8/10 (80%)  | 27,944 | 22,939 | 42,066 |
| 0.5               | 9/10 (90%)  | 19,528 | 13,949 | 33,499 |
| 0.7               | 9/10 (90%)  | 10,479 | 7,910  | 12,956 |
| 0.9               | 7/10 (70%)  | 9,343  | 3,678  | 25,289 |

![モメンタム比較](Practice/momentum_comparison.png)

*図1: 左: alpha = 0.0, 0.5, 0.9 の誤差曲線（seed = 1）。右: モメンタム係数別の平均エポック数と収束率。*

一番バランスが良かったのはalpha = 0.5〜0.7あたりだった。alpha = 0.7では平均エポック数が約10,500まで下がり、ベースラインの3分の1ほどになった。収束率も60%から90%に改善している。速度の改善もさることながら、モメンタムなしでは局所解にはまっていた試行が収束するようになったことのほうが実用上は大きいと思う。

alpha = 0.9では、収束した試行の速度は最速（平均~9,300、最良で3,678エポック）だったが、収束率が70%に落ちた。中には正答率11/16でランダムより悪い結果に終わった試行もあった。モメンタムが大きすぎると蓄積された速度が誤差曲面の良い領域を通り過ぎてしまう。坂をボールで押しすぎて谷底を飛び越えてしまうようなイメージだ。

---

## f) 拡張実験の考察

### 8ビットパリティ: 難易度のスケーリング

8ビットの実験で、パリティ問題の難しさがどれだけ急激にスケールするかがはっきりした。4ビット（16パターン）は通常のBPで数万エポックかければ解けたが、8ビット（256パターン）は隠れ32個・10万エポックをかけても収束しなかった。

これは単にデータ量が増えたからではない。パリティ関数には「どの入力ビット1つを反転しても出力が変わる」という性質がある。つまり入力空間のどの方向に見ても、0と1が交互に並んでいて局所的な構造がまったくない。4ビットのときは16行の真理値表を覚えればよく、各隠れニューロンが入力空間をうまく分割できた。しかし8ビットでは256行あり、関数が「最大限に非局所的」になっている。ネットワークは真理値表を丸暗記する必要があるが、誤差曲面は勾配がほぼゼロの広い平坦領域だらけになり、通常のBPではそこから抜け出せない。ほとんどの試行が227/256（89%）で止まったのは、部分的な解は見つけたが最後の微調整ができなかったということだろう。

### モメンタム: なぜ効くか、そしていつ裏目に出るか

モメンタムの実験からは、明確なトレードオフが見えた。適度なモメンタム（0.5〜0.7）は収束を速めるだけでなく局所解からの脱出にも効く。しかしモメンタムが大きすぎる（0.9）と不安定になる。

もう少し定量的に考えてみる。更新則を再掲すると:

```
Delta_w(t) = eta * g(t) + alpha * Delta_w(t-1)
```

ここで `g(t) = delta * y` は勾配項。もし勾配が一定（毎ステップ同じ `g`）ならモメンタムの定常状態での実効学習率は `eta / (1 - alpha)` になる。alpha = 0.9だと `0.5 / 0.1 = 5.0` で名目の10倍。収束した試行が速いのはこのためだ。

しかし勾配は一定ではない。各パターンごとに違う勾配が出るし、重みが変われば勾配も変わる。勾配の分散が大きいと（方向がバラバラだと）、モメンタム項がそのノイズを増幅する。alpha = 0.9で失敗した試行では重みが暴れて、11/16正答というランダム以下の結果になった。これはまさにモメンタムがノイズ増幅器として働いてしまった例だ。

### 2つの実験をつなげて

8ビットパリティ問題は、実はモメンタムが最も効きそうなタイプの問題でもある。誤差曲面には勾配が小さいが一貫した方向を向いている長い平坦領域がある。通常のBPはこういう場所でジリジリとしか進めないが、モメンタムがあれば速度が蓄積して平坦領域を素早く横断できる。大きな隠れ層（32個以上）と適度なモメンタム（0.5〜0.7）を組み合わせ、さらにエポック数を増やせば8ビットパリティも最終的には解けるだろうと予想している。ただし、4ビットよりはるかに時間がかかることは間違いない。

総合すると、この2つの実験はBPの限界の異なる側面を示している。8ビット実験は「問題の複雑さがネットワークの容量を超えうること」を、モメンタム実験は「同じネットワーク構造でもアルゴリズムの改良で学習を大きく加速できること」を示した。実用的には、ネットワークの適切なサイズ選択と最適化手法のチューニング、両方が等しく重要だということだ。

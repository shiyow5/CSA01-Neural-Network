# CSA01 - チームプロジェクト1 パート1: 単一ニューロンの学習

**科目:** ニューラルネットワーク (CSA01)
**チームメンバー:**
- 佐藤 丞 (m5301059)
- 宇佐美 雄貴 (m5301073)
- 関根 健人 (m5301060)
- 相澤 祐真 (m5301001)
- 渡部 千歳 (m5301074)

---

## a) 解いた問題

パート1では、単一ニューロンに対するパーセプトロン学習則とデルタ学習則の2つを実装し、動作の違いを比較した。

対象問題には **AND ゲート** を選んだ。2つの二値入力を受け取り、両方が1のときだけ+1を出力する関数で、線形分離可能なため単一ニューロンで学習できる。訓練データは以下の通り（3番目の入力はバイアス用の-1）:

| 入力 x1 | 入力 x2 | バイアス | 期待出力 d |
|:-------:|:-------:|:-------:|:---------:|
| 0       | 0       | -1      | -1        |
| 0       | 1       | -1      | -1        |
| 1       | 0       | -1      | -1        |
| 1       | 1       | -1      | 1         |

### ソースコード: パーセプトロン学習則 (`perceptron_learning.c`)

```c
/*************************************************************/
/* C-program for perceptron-learning rule                    */
/* Learning rule of one neuron                               */
/*                                                           */
/* This program is produced by m5301059 SATO Sho.            */
/*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I 3
#define n_sample 4
#define eta 0.5
#define lambda 1.0
#define desired_error 0.01
#define stepf(x) (x >= 0 ? 1 : -1)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

// 2d { x, y, dummy input for bias(=-1)}
double x[n_sample][I] = {
    {0, 0, -1},
    {0, 1, -1},
    {1, 0, -1},
    {1, 1, -1},
};

double w[I];
double d[n_sample] = {-1, -1, -1, 1};
double o;

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main()
{
  int i, p, q = 0;
  double delta, Error = DBL_MAX, LearningSignal;

  Initialization();
  while (Error > desired_error)
  {
    q++;
    Error = 0;
    for (p = 0; p < n_sample; p++)
    {
      FindOutput(p);
      Error += 0.5 * pow(d[p] - o, 2.0);
      LearningSignal = eta * (d[p] - o);
      for (i = 0; i < I; i++)
      {
        w[i] += LearningSignal * x[p][i];
      }
      printf("Error in the %d-th learning cycle=%f\n", q, Error);
    }
  }
  PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
  int i;

  randomize();
  for (i = 0; i < I; i++)
    w[i] = frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p)
{
  int i;
  double temp = 0;

  for (i = 0; i < I; i++)
    temp += w[i] * x[p][i];
  o = stepf(temp);
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void)
{
  int i, p;

  printf("\n\n");
  printf("The connection weights of the neurons:\n");
  for (i = 0; i < I; i++)
    printf("%5f ", w[i]);
  printf("\n\n");

  printf("Neuron output for each input pattern:\n");
  for (p = 0; p < n_sample; p++)
  {
    FindOutput(p);
    printf("(");
    for (i = 0; i < I; i++)
      printf(" %.1f,", x[p][i]);
    printf(") -> %5f\n", o);
  }
  printf("\n");
}
```

### ソースコード: デルタ学習則 (`delta_learning.c`)

```c
/*********************************************************************************/
/* C-program for delta-learning rule                                             */
/* Learning rule of one neuron                                                   */
/*                                                                               */
/* This program is produced by Qiangfu Zhao and extended by m5301059 SATO Sho.   */
/* You are free to use it for educational purpose                                */
/*********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I 3
#define n_sample 4
#define eta 0.5
#define lambda 1.0
#define desired_error 0.01
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

// 2d { x, y, dummy input for bias(=-1)}
double x[n_sample][I] = {
    {0, 0, -1},
    {0, 1, -1},
    {1, 0, -1},
    {1, 1, -1},
};

double w[I];
double d[n_sample] = {-1, -1, -1, 1};
double o;

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main()
{
    int i, p, q = 0;
    double delta, Error = DBL_MAX;

    Initialization();
    while (Error > desired_error)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            FindOutput(p);
            Error += 0.5 * pow(d[p] - o, 2.0);
            for (i = 0; i < I; i++)
            {
                delta = (d[p] - o) * (1 - o * o) / 2;
                w[i] += eta * delta * x[p][i];
            }
            printf("Error in the %d-th learning cycle=%f\n", q, Error);
        }
    }
    PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
    int i;

    randomize();
    for (i = 0; i < I; i++)
        w[i] = frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p)
{
    int i;
    double temp = 0;

    for (i = 0; i < I; i++)
        temp += w[i] * x[p][i];
    o = sigmoid(temp);
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void)
{
    int i, p;

    printf("\n\n");
    printf("The connection weights of the neurons:\n");
    for (i = 0; i < I; i++)
        printf("%5f ", w[i]);
    printf("\n\n");

    printf("Neuron output for each input pattern:\n");
    for (p = 0; p < n_sample; p++)
    {
        FindOutput(p);
        printf("(");
        for (i = 0; i < I; i++)
            printf(" %.1f,", x[p][i]);
        printf(") -> %5f\n", o);
    }
    printf("\n");
}
```

## b) 使用した手法

### パーセプトロン学習則

活性化関数にステップ関数を使う:

```
f(x) = +1  (x >= 0 のとき)
f(x) = -1  (x < 0 のとき)
```

パターンを提示するたびに重みを更新する:

```
w_i(t+1) = w_i(t) + eta * (d - o) * x_i
```

`eta = 0.5` が学習率、`d` が期待出力、`o` が実際の出力。誤差は `E = 0.5 * (d - o)^2` で計算し、4サンプル全体の合計誤差が 0.01 を下回ったら学習終了とする。

### デルタ学習則

活性化関数に [-1, 1] にスケーリングしたシグモイド関数を使う:

```
f(x) = 2 / (1 + exp(-lambda * x)) - 1,  lambda = 1.0
```

重みの更新にはシグモイドの導関数を利用する:

```
delta = (d - o) * (1 - o^2) / 2
w_i(t+1) = w_i(t) + eta * delta * x_i
```

収束条件はパーセプトロンと同じ（合計誤差 < 0.01）。

### 共通の設定

- 入力数: 3（データ2つ + バイアス1つ）
- 学習率: 0.5
- 重みは [0, 1) でランダム初期化

## c) シミュレーション結果に関する考察

### パーセプトロンの結果

**5エポック**で収束した。最終的な重み:

```
w = [1.528, 1.227, 2.427]
```

| 入力         | 出力  | 期待値 |
|:-----------:|:-----:|:-----:|
| (0, 0, -1)  | -1    | -1    |
| (0, 1, -1)  | -1    | -1    |
| (1, 0, -1)  | -1    | -1    |
| (1, 1, -1)  | +1    | +1    |

全パターンで正しく分類できた。

### デルタ学習の結果

収束まで **969エポック** かかった。最終的な重み:

```
w = [6.281, 6.278, 9.503]
```

| 入力         | 出力      | 期待値 |
|:-----------:|:--------:|:-----:|
| (0, 0, -1)  | -0.9999  | -1    |
| (0, 1, -1)  | -0.9235  | -1    |
| (1, 0, -1)  | -0.9233  | -1    |
| (1, 1, -1)  | +0.9101  | +1    |

出力は期待値に近いが、ぴったりとは一致しない。シグモイド関数は+/-1に漸近するだけで到達しないので、これは仕方がない。

### 比較

一番目立つ違いは収束の速さで、パーセプトロンが5エポックなのに対してデルタ則は969エポックもかかった。パーセプトロンはステップ関数でぴったり+1/-1を出すので、決定境界が正しい位置に来ればすぐに誤差がゼロになる。一方、デルタ則のシグモイド出力は+/-1に少しずつ近づくだけなので、重みを大きくし続けないと誤差が下がらない。実際、デルタ則の最終重みは6〜9と、パーセプトロンの1〜2に比べてかなり大きくなった。

誤差の減り方にも違いがあった。パーセプトロンの誤差は2の倍数で飛び飛びに変化する（出力が間違うと `(d-o)^2 = 4` になるため）のに対し、デルタ則の誤差は滑らかに減少していく。

ただし、デルタ則には微分可能な活性化関数を使っているという利点がある。ANDのような単純な問題では違いが出ないが、多層ネットワークを誤差逆伝播法で学習させるには微分可能性が必要になるので、そこでデルタ則の考え方が活きてくる。

---

## d) 新しい問題: XOR ゲート

線形分離できない問題に対してどうなるかを確認するため、**XOR問題**で両学習則を試してみた。

| 入力 x1 | 入力 x2 | 期待出力 d |
|:-------:|:-------:|:---------:|
| 0       | 0       | -1        |
| 0       | 1       | +1        |
| 1       | 0       | +1        |
| 1       | 1       | -1        |

最大1,000エポックまで学習させたが、どちらも収束しなかった:

- **パーセプトロン:** 誤差が8.0のまま変わらなかった。4パターン中2パターンが常に誤分類されて、各4.0の誤差が出続ける。
- **デルタ則:** 誤差が2.59あたりで止まった。出力が全部0付近に寄ってしまい、+1にも-1にもならない状態になった。

これは予想通りの結果で、単一ニューロンは `w1*x1 + w2*x2 = 閾値` という直線1本でしか分類できない。XORでは+1のクラス（(0,1)と(1,0)）と-1のクラス（(0,0)と(1,1)）が対角に配置されているので、どう直線を引いても全部正しくは分けられない。

![XOR 誤差曲線](Practice/Part1/xor_error.png)

*図1: XOR問題の誤差推移。パーセプトロンは8.0で横ばい、デルタ則は2.6付近で停滞している。*

---

## e) 手法の改良: 学習率の比較

学習率 `eta` を変えるとAND問題の収束がどう変わるかを調べた。0.01, 0.1, 0.5, 1.0, 2.0, 5.0 の6通りを、それぞれランダムシードを変えて10回ずつ試した。

| eta  | パーセプトロン平均エポック | パーセプトロン収束率 | デルタ平均エポック | デルタ収束率 |
|:----:|:---------------------:|:------------------:|:----------------:|:-----------:|
| 0.01 | 9.5                   | 100%               | N/A              | 0%          |
| 0.10 | 3.5                   | 100%               | 4947.5           | 100%        |
| 0.50 | 6.1                   | 100%               | 971.6            | 100%        |
| 1.00 | 6.7                   | 100%               | 474.5            | 100%        |
| 2.00 | 6.7                   | 100%               | 225.1            | 100%        |
| 5.00 | 6.7                   | 100%               | 29.6             | 100%        |

パーセプトロンはどの学習率でも問題なく収束した。eta による違いもほとんどない。

一方、デルタ則では学習率の影響が非常に大きかった。eta = 0.01 では10,000エポック以内に収束できず、eta = 5.0 だと約30エポックで終わった。170倍近い差がある。これはおそらく、シグモイドの導関数 `(1-o^2)/2` が出力の絶対値が大きいところで非常に小さくなるためで、そこに小さい eta が掛け合わさると重み更新がほぼゼロになってしまう。eta を大きくすることでこれを補える。ただし、もっと複雑な問題では eta が大きすぎると発散する恐れがあるので、今回のAND問題は単純なケースと言える。

![学習率比較](Practice/Part1/eta_comparison.png)

*図2: 学習率比較。上段: 誤差曲線。左下: 平均収束エポック数（対数スケール）。右下: 収束率。*

---

## f) 拡張実験に関する考察

XOR実験では、単一ニューロンでは線形分離できない問題が解けないことを実際に確認できた。どの学習則を使っても誤差が下がらないので、こういう問題を解くには多層ネットワークが必要になる。

学習率の実験では、パーセプトロンは eta にほとんど影響されないのに対し、デルタ則は eta の値で収束速度が大きく変わることがわかった。勾配ベースの手法では学習率の選び方が重要だということが、この簡単な実験からもよく見て取れる。

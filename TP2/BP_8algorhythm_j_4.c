/*************************************************************/
/* C-program for BP algorithm (8-bit Parity Check)           */
/*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I             9   /* 8ビットの入力 + 1つのダミー */
#define J             5  /* 隠れ層10個 + 1つのダミー（課題に合わせて調整可） */
#define K             1   /* 出力層 */
#define n_sample      256 /* 8ビットの全組み合わせ数 (2^8) */
#define eta           0.5
#define lambda        1.0
#define desired_error 0.001
#define sigmoid(a)    (1.0/(1.0+exp(-lambda*(a)))) /* 引数名をxからaに変更 */
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))

/* 8ビットの入出力パターン用変数 */
double x[n_sample][I];
double d[n_sample][K];

double v[J][I], w[K][J];
double y[J];
double o[K];

void Initialization(void);
void FindHidden(int p);
void FindOutput(void);
void PrintResult(void);

int main() {
  int i, j, k, p, q = 0;
  double Error = DBL_MAX;
  double delta_o[K];
  double delta_y[J];

  /* 8ビットのデータ生成 (元のコードの直書き部分をループに置換) */
  for (p = 0; p < n_sample; p++) {
    int count = 0;
    for (i = 0; i < 8; i++) {
      x[p][i] = (p >> i) & 1; // pのiビット目を取り出す
      if (x[p][i] == 1) count++;
    }
    x[p][8] = -1; // ダミー入力
    d[p][0] = (count % 2 == 0) ? 1 : 0; // 偶数なら1、奇数なら0
  }

  Initialization();
  
  while (Error > desired_error && q < 2000000) { // 8ビットは学習に時間がかかるため上限増加
    q++;
    Error = 0;
    for (p = 0; p < n_sample; p++) {
      FindHidden(p);
      FindOutput();

      for (k = 0; k < K; k++) {
        Error += 0.5 * pow(d[p][k] - o[k], 2.0);
        delta_o[k] = (d[p][k] - o[k]) * (1 - o[k]) * o[k];
      }
      
      for (j = 0; j < J; j++) {
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
    
    if (q % 2000 == 0) {
      printf("Error in the %d-th learning cycle = %f\n", q, Error);
    }
  } 
  
  printf("\nFinished at the %d-th learning cycle with Error = %f\n", q, Error);

  PrintResult();
  return 0;
}
  
/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void) {
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
void FindHidden(int p) {
  int i, j;
  double temp;
  for (j = 0; j < J - 1; j++) {
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
void FindOutput(void) {
  int j, k;
  double temp;
  for (k = 0; k < K; k++) {
    temp = 0;
    for (j = 0; j < J; j++)
      temp += w[k][j] * y[j];
    o[k] = sigmoid(temp);
  }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void) {
  int j, k;
  printf("\nFinished. Weights summarized in internal variables.\n");
  /* 8ビットの場合、重みの出力が膨大になるため、代表して出力層のみ表示 */
  printf("The connection weights in the output layer:\n");
  for (k = 0; k < K; k++) {
    for (j = 0; j < J; j++)
      printf("%5f ", w[k][j]);
    printf("\n");
  }
}

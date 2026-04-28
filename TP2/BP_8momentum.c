/*************************************************************/
/* C-program for BP with Momentum (8-bit Parity Check)       */
/* Compile: gcc -O2 -o momentum BP_8momentum.c -lm           */
/* Usage:   ./momentum <hidden_neurons>                      */
/*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I             9
#define MAX_J         30
#define K             1
#define n_sample      256
#define eta           0.5
#define alpha         0.9
#define lambda        1.0
#define desired_error 0.001
#define sigmoid(a)    (1.0/(1.0+exp(-lambda*(a))))
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))

int J; /* set from command line: hidden_neurons + 1 */

double x[n_sample][I];
double d[n_sample][K];
double v[MAX_J][I], w[K][MAX_J];
double dv[MAX_J][I], dw[K][MAX_J];
double y[MAX_J], o[K];

int main(int argc, char *argv[]) {
  int i, j, k, p, q = 0;
  double Error = DBL_MAX;
  double delta_o[K], delta_y[MAX_J], change;

  if (argc < 2) { printf("Usage: %s <hidden_neurons>\n", argv[0]); return 1; }
  int hidden = atoi(argv[1]);
  J = hidden + 1;

  for (p = 0; p < n_sample; p++) {
    int count = 0;
    for (i = 0; i < 8; i++) {
      x[p][i] = (p >> i) & 1;
      if (x[p][i] == 1) count++;
    }
    x[p][8] = -1;
    d[p][0] = (count % 2 == 0) ? 1 : 0;
  }

  randomize();
  for (j = 0; j < J; j++)
    for (i = 0; i < I; i++) { v[j][i] = frand() - 0.5; dv[j][i] = 0.0; }
  for (k = 0; k < K; k++)
    for (j = 0; j < J; j++) { w[k][j] = frand() - 0.5; dw[k][j] = 0.0; }

  while (Error > desired_error && q < 2000000) {
    q++;
    Error = 0;
    for (p = 0; p < n_sample; p++) {
      /* FindHidden */
      for (j = 0; j < J - 1; j++) {
        double temp = 0;
        for (i = 0; i < I; i++) temp += v[j][i] * x[p][i];
        y[j] = sigmoid(temp);
      }
      y[J - 1] = -1;

      /* FindOutput */
      for (k = 0; k < K; k++) {
        double temp = 0;
        for (j = 0; j < J; j++) temp += w[k][j] * y[j];
        o[k] = sigmoid(temp);
      }

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
        for (j = 0; j < J; j++) {
          change = eta * delta_o[k] * y[j] + alpha * dw[k][j];
          w[k][j] += change;
          dw[k][j] = change;
        }
      for (j = 0; j < J; j++)
        for (i = 0; i < I; i++) {
          change = eta * delta_y[j] * x[p][i] + alpha * dv[j][i];
          v[j][i] += change;
          dv[j][i] = change;
        }
    }
    if (q % 2000 == 0)
      printf("Error in the %d-th learning cycle = %f\n", q, Error);
  }
  printf("\nFinished at the %d-th learning cycle with Error = %f\n", q, Error);
  return 0;
}

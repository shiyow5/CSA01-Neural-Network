
/***********************************************************************************/
/* C-program for learning of single layer neural network                           */
/* based on the delta learning rule                                                */
/*                                                                                 */
/*  1) Number of Inputs : N                                                        */
/*  2) Number of Output : R                                                        */
/* The last input for all neurons is always -1                                     */
/*                                                                                 */
/* This program is produced by Qiangfu Zhao and extended by m5301059 SATO Sho.     */
/* You are free to use it for educational purpose                                  */
/***********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 3
#define R 3
#define n_sample 3
#define eta 0.5
#define lambda 1.0
#define desired_error 0.1
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

double x[n_sample][N] = {
    {10, 2, -1},
    {2, -5, -1},
    {-5, 5, -1},
};
double d[n_sample][R] = {
    {1, -1, -1},
    {-1, 1, -1},
    {-1, -1, 1},
};
double w[R][N];
double o[R];

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main()
{
  int i, j, p, q = 0;
  double Error = DBL_MAX;
  double delta;

  Initialization();
  while (Error > desired_error)
  {
    q++;
    Error = 0;
    for (p = 0; p < n_sample; p++)
    {
      FindOutput(p);
      for (i = 0; i < R; i++)
      {
        Error += 0.5 * pow(d[p][i] - o[i], 2.0);
      }
      for (i = 0; i < R; i++)
      {
        delta = (d[p][i] - o[i]) * (1 - o[i] * o[i]) / 2;
        for (j = 0; j < N; j++)
        {
          w[i][j] += eta * delta * x[p][j];
        }
      }
    }
    printf("Error in the %d-th learning cycle=%f\n", q, Error);
  }
  PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
  int i, j;

  randomize();
  for (i = 0; i < R; i++)
    for (j = 0; j < N; j++)
      w[i][j] = frand() - 0.5;
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p)
{
  int i, j;
  double temp;

  for (i = 0; i < R; i++)
  {
    temp = 0;
    for (j = 0; j < N; j++)
    {
      temp += w[i][j] * x[p][j];
    }
    o[i] = sigmoid(temp);
  }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void)
{
  int i, j, p;

  printf("\n\n");
  printf("The connection weights are:\n");
  for (i = 0; i < R; i++)
  {
    for (j = 0; j < N; j++)
      printf("%5f ", w[i][j]);
    printf("\n");
  }
  printf("\n\n");

    printf("Neuron output for each input pattern:\n");
  for (p = 0; p < n_sample; p++)
  {
    FindOutput(p);
    printf("(");
    for (i = 0; i < N; i++)
      printf(" %.1f,", x[p][i]);
    printf(") -> (");
    for (i = 0; i < R; i++)
      printf(" %5f,", o[i]);
    printf(")\n");
  }
  printf("\n");
}

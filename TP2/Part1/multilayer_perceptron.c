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

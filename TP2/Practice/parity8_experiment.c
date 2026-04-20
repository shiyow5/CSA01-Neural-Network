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

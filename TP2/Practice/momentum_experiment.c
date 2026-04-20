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
    FILE *fp = NULL;

    /* If seed == 1 and DUMP_CSV is defined, output error curve */
#ifdef DUMP_CSV
    char filename[256];
    sprintf(filename, "momentum_a%d_j%d.csv", ALPHA_VAL, J_VAL);
    fp = fopen(filename, "w");
    fprintf(fp, "epoch,error\n");
#endif

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

        if (fp) fprintf(fp, "%d,%f\n", q, Error);
    }

    if (fp) fclose(fp);

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

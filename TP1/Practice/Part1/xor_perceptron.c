/*************************************************************/
/* XOR problem with perceptron learning rule                 */
/* Demonstrates that a single neuron cannot solve XOR        */
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
#define MAX_EPOCH 1000
#define desired_error 0.01
#define stepf(x) (x >= 0 ? 1 : -1)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

double x[n_sample][I] = {
    {0, 0, -1},
    {0, 1, -1},
    {1, 0, -1},
    {1, 1, -1},
};

double w[I];
double d[n_sample] = {-1, 1, 1, -1}; /* XOR */
double o;

void Initialization(void)
{
    int i;
    randomize();
    for (i = 0; i < I; i++)
        w[i] = frand();
}

void FindOutput(int p)
{
    int i;
    double temp = 0;
    for (i = 0; i < I; i++)
        temp += w[i] * x[p][i];
    o = stepf(temp);
}

int main()
{
    int i, p, q = 0;
    double Error = DBL_MAX, LearningSignal;
    FILE *fp = fopen("xor_perceptron_error.csv", "w");

    fprintf(fp, "epoch,error\n");
    Initialization();

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            FindOutput(p);
            Error += 0.5 * pow(d[p] - o, 2.0);
            LearningSignal = eta * (d[p] - o);
            for (i = 0; i < I; i++)
                w[i] += LearningSignal * x[p][i];
        }
        fprintf(fp, "%d,%f\n", q, Error);
    }
    fclose(fp);

    printf("Perceptron XOR: %s after %d epochs (final error=%f)\n",
           (Error <= desired_error) ? "CONVERGED" : "DID NOT CONVERGE", q, Error);

    printf("Final outputs:\n");
    for (p = 0; p < n_sample; p++)
    {
        FindOutput(p);
        printf("  (%0.f, %0.f) -> %.0f (desired: %.0f)\n",
               x[p][0], x[p][1], o, d[p]);
    }
    return 0;
}

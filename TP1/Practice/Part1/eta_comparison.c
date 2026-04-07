/*************************************************************/
/* Learning rate comparison for AND gate                     */
/* Compares perceptron and delta learning with various eta   */
/*                                                           */
/* Usage: ./eta_comparison <eta_value>                       */
/* This program is produced by m5301059 SATO Sho.            */
/*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I 3
#define n_sample 4
#define lambda 1.0
#define MAX_EPOCH 10000
#define desired_error 0.01
#define stepf(x) (x >= 0 ? 1 : -1)
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)

double x[n_sample][I] = {
    {0, 0, -1},
    {0, 1, -1},
    {1, 0, -1},
    {1, 1, -1},
};
double d[n_sample] = {-1, -1, -1, 1}; /* AND */

int run_perceptron(double eta, unsigned int seed, FILE *fp)
{
    double w[I], o;
    int i, p, q = 0;
    double Error = DBL_MAX, LearningSignal;

    srand(seed);
    for (i = 0; i < I; i++) w[i] = frand();

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            double temp = 0;
            for (i = 0; i < I; i++) temp += w[i] * x[p][i];
            o = stepf(temp);
            Error += 0.5 * pow(d[p] - o, 2.0);
            LearningSignal = eta * (d[p] - o);
            for (i = 0; i < I; i++) w[i] += LearningSignal * x[p][i];
        }
        if (fp) fprintf(fp, "%d,%f\n", q, Error);
    }
    return (Error <= desired_error) ? q : -1;
}

int run_delta(double eta, unsigned int seed, FILE *fp)
{
    double w[I], o;
    int i, p, q = 0;
    double delta, Error = DBL_MAX;

    srand(seed);
    for (i = 0; i < I; i++) w[i] = frand();

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            double temp = 0;
            for (i = 0; i < I; i++) temp += w[i] * x[p][i];
            o = sigmoid(temp);
            Error += 0.5 * pow(d[p] - o, 2.0);
            for (i = 0; i < I; i++)
            {
                delta = (d[p] - o) * (1 - o * o) / 2;
                w[i] += eta * delta * x[p][i];
            }
        }
        if (fp) fprintf(fp, "%d,%f\n", q, Error);
    }
    return (Error <= desired_error) ? q : -1;
}

int main()
{
    double etas[] = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0};
    int n_etas = 6;
    int n_trials = 10;
    int e, t;
    unsigned int base_seed = 42;
    FILE *fp;
    char filename[256];

    /* Summary CSV */
    fp = fopen("eta_summary.csv", "w");
    fprintf(fp, "eta,method,avg_epochs,min_epochs,max_epochs,convergence_rate\n");

    for (e = 0; e < n_etas; e++)
    {
        int p_total = 0, p_min = MAX_EPOCH + 1, p_max = 0, p_conv = 0;
        int d_total = 0, d_min = MAX_EPOCH + 1, d_max = 0, d_conv = 0;

        for (t = 0; t < n_trials; t++)
        {
            unsigned int seed = base_seed + t;
            int ep;

            ep = run_perceptron(etas[e], seed, NULL);
            if (ep > 0) { p_total += ep; p_conv++; if (ep < p_min) p_min = ep; if (ep > p_max) p_max = ep; }

            ep = run_delta(etas[e], seed, NULL);
            if (ep > 0) { d_total += ep; d_conv++; if (ep < d_min) d_min = ep; if (ep > d_max) d_max = ep; }
        }

        fprintf(fp, "%.2f,perceptron,%.1f,%d,%d,%.0f%%\n",
                etas[e],
                p_conv > 0 ? (double)p_total / p_conv : -1.0,
                p_conv > 0 ? p_min : -1,
                p_conv > 0 ? p_max : -1,
                100.0 * p_conv / n_trials);
        fprintf(fp, "%.2f,delta,%.1f,%d,%d,%.0f%%\n",
                etas[e],
                d_conv > 0 ? (double)d_total / d_conv : -1.0,
                d_conv > 0 ? d_min : -1,
                d_conv > 0 ? d_max : -1,
                100.0 * d_conv / n_trials);

        printf("eta=%.2f  Perceptron: avg=%.1f (conv=%d/%d)  Delta: avg=%.1f (conv=%d/%d)\n",
               etas[e],
               p_conv > 0 ? (double)p_total / p_conv : -1.0, p_conv, n_trials,
               d_conv > 0 ? (double)d_total / d_conv : -1.0, d_conv, n_trials);
    }
    fclose(fp);

    /* Detailed error curves for selected etas */
    double detail_etas[] = {0.1, 0.5, 1.0, 2.0};
    int n_detail = 4;
    for (e = 0; e < n_detail; e++)
    {
        sprintf(filename, "eta_perceptron_%.2f.csv", detail_etas[e]);
        fp = fopen(filename, "w");
        fprintf(fp, "epoch,error\n");
        run_perceptron(detail_etas[e], 42, fp);
        fclose(fp);

        sprintf(filename, "eta_delta_%.2f.csv", detail_etas[e]);
        fp = fopen(filename, "w");
        fprintf(fp, "epoch,error\n");
        run_delta(detail_etas[e], 42, fp);
        fclose(fp);
    }

    printf("\nCSV files generated.\n");
    return 0;
}

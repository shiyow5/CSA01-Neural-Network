/*************************************************************/
/* Weight initialization range influence                     */
/* Tests different init ranges and seeds for convergence     */
/*                                                           */
/* This program is produced by m5301059 SATO Sho.            */
/*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 3
#define R 3
#define n_sample 3
#define MAX_EPOCH 50000
#define lambda 1.0
#define eta 0.5
#define desired_error 0.1
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)

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

int run_delta(double init_range, unsigned int seed, FILE *fp)
{
    double w[R][N], o[R];
    int i, j, p, q = 0;
    double Error = DBL_MAX, delta;

    srand(seed);
    for (i = 0; i < R; i++)
        for (j = 0; j < N; j++)
            w[i][j] = (frand() - 0.5) * 2.0 * init_range;

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            for (i = 0; i < R; i++)
            {
                double temp = 0;
                for (j = 0; j < N; j++) temp += w[i][j] * x[p][j];
                o[i] = sigmoid(temp);
            }
            for (i = 0; i < R; i++)
                Error += 0.5 * pow(d[p][i] - o[i], 2.0);
            for (i = 0; i < R; i++)
            {
                delta = (d[p][i] - o[i]) * (1 - o[i] * o[i]) / 2;
                for (j = 0; j < N; j++)
                    w[i][j] += eta * delta * x[p][j];
            }
        }
        if (fp) fprintf(fp, "%d,%f\n", q, Error);
    }
    return (Error <= desired_error) ? q : -1;
}

int run_perceptron(double init_range, unsigned int seed, FILE *fp)
{
    double w[R][N], o[R];
    int i, j, p, q = 0;
    double Error = DBL_MAX;

    srand(seed);
    for (i = 0; i < R; i++)
        for (j = 0; j < N; j++)
            w[i][j] = (frand() - 0.5) * 2.0 * init_range;

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            for (i = 0; i < R; i++)
            {
                double temp = 0;
                for (j = 0; j < N; j++) temp += w[i][j] * x[p][j];
                o[i] = (temp >= 0) ? 1 : -1;
            }
            for (i = 0; i < R; i++)
                Error += 0.5 * pow(d[p][i] - o[i], 2.0);
            for (i = 0; i < R; i++)
                for (j = 0; j < N; j++)
                    w[i][j] += eta * (d[p][i] - o[i]) * x[p][j];
        }
        if (fp) fprintf(fp, "%d,%f\n", q, Error);
    }
    return (Error <= desired_error) ? q : -1;
}

int main()
{
    double ranges[] = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0};
    int n_ranges = 6;
    int n_trials = 20;
    unsigned int base_seed = 42;
    int r, t;
    FILE *fp;
    char filename[256];

    fp = fopen("weight_init_summary.csv", "w");
    fprintf(fp, "init_range,method,avg_epochs,min_epochs,max_epochs,std_epochs,convergence_rate\n");

    for (r = 0; r < n_ranges; r++)
    {
        int d_epochs[20], p_epochs[20];
        int d_conv = 0, p_conv = 0;
        double d_sum = 0, p_sum = 0, d_sq = 0, p_sq = 0;

        for (t = 0; t < n_trials; t++)
        {
            unsigned int seed = base_seed + t;

            d_epochs[t] = run_delta(ranges[r], seed, NULL);
            if (d_epochs[t] > 0) { d_sum += d_epochs[t]; d_sq += d_epochs[t] * d_epochs[t]; d_conv++; }

            p_epochs[t] = run_perceptron(ranges[r], seed, NULL);
            if (p_epochs[t] > 0) { p_sum += p_epochs[t]; p_sq += p_epochs[t] * p_epochs[t]; p_conv++; }
        }

        double d_avg = d_conv > 0 ? d_sum / d_conv : -1;
        double d_std = d_conv > 1 ? sqrt((d_sq - d_sum * d_sum / d_conv) / (d_conv - 1)) : 0;
        double p_avg = p_conv > 0 ? p_sum / p_conv : -1;
        double p_std = p_conv > 1 ? sqrt((p_sq - p_sum * p_sum / p_conv) / (p_conv - 1)) : 0;

        int d_min = MAX_EPOCH + 1, d_max = 0, p_min = MAX_EPOCH + 1, p_max = 0;
        for (t = 0; t < n_trials; t++) {
            if (d_epochs[t] > 0) { if (d_epochs[t] < d_min) d_min = d_epochs[t]; if (d_epochs[t] > d_max) d_max = d_epochs[t]; }
            if (p_epochs[t] > 0) { if (p_epochs[t] < p_min) p_min = p_epochs[t]; if (p_epochs[t] > p_max) p_max = p_epochs[t]; }
        }

        fprintf(fp, "%.2f,delta,%.1f,%d,%d,%.1f,%.0f%%\n",
                ranges[r], d_avg, d_conv > 0 ? d_min : -1, d_conv > 0 ? d_max : -1, d_std, 100.0 * d_conv / n_trials);
        fprintf(fp, "%.2f,perceptron,%.1f,%d,%d,%.1f,%.0f%%\n",
                ranges[r], p_avg, p_conv > 0 ? p_min : -1, p_conv > 0 ? p_max : -1, p_std, 100.0 * p_conv / n_trials);

        printf("range=%.2f  Delta: avg=%.1f std=%.1f (conv=%d/%d)  Perceptron: avg=%.1f std=%.1f (conv=%d/%d)\n",
               ranges[r], d_avg, d_std, d_conv, n_trials, p_avg, p_std, p_conv, n_trials);
    }
    fclose(fp);

    /* Detailed error curves for selected ranges */
    double detail_ranges[] = {0.01, 0.1, 0.5, 2.0, 5.0};
    int n_detail = 5;
    for (r = 0; r < n_detail; r++)
    {
        sprintf(filename, "winit_delta_%.2f.csv", detail_ranges[r]);
        fp = fopen(filename, "w");
        fprintf(fp, "epoch,error\n");
        run_delta(detail_ranges[r], 42, fp);
        fclose(fp);
    }

    printf("\nCSV files generated.\n");
    return 0;
}

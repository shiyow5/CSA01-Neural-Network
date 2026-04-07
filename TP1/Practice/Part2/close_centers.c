/*************************************************************/
/* 3-class classification with closer class centers          */
/* Compares convergence difficulty with original problem      */
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
#define MAX_EPOCH 50000
#define lambda 1.0
#define desired_error 0.1
#define stepf(x) (x >= 0 ? 1 : -1)
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)

typedef struct {
    const char *name;
    double x[3][N];
    double d[3][R];
} Dataset;

int run_perceptron(Dataset *ds, double eta, unsigned int seed, FILE *fp)
{
    double w[R][N], o[R];
    int i, j, p, q = 0;
    double Error = DBL_MAX;

    srand(seed);
    for (i = 0; i < R; i++)
        for (j = 0; j < N; j++)
            w[i][j] = frand() - 0.5;

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < 3; p++)
        {
            for (i = 0; i < R; i++)
            {
                double temp = 0;
                for (j = 0; j < N; j++) temp += w[i][j] * ds->x[p][j];
                o[i] = stepf(temp);
            }
            for (i = 0; i < R; i++)
                Error += 0.5 * pow(ds->d[p][i] - o[i], 2.0);
            for (i = 0; i < R; i++)
                for (j = 0; j < N; j++)
                    w[i][j] += eta * (ds->d[p][i] - o[i]) * ds->x[p][j];
        }
        if (fp) fprintf(fp, "%d,%f\n", q, Error);
    }
    return (Error <= desired_error) ? q : -1;
}

int run_delta(Dataset *ds, double eta, unsigned int seed, FILE *fp)
{
    double w[R][N], o[R];
    int i, j, p, q = 0;
    double Error = DBL_MAX, delta;

    srand(seed);
    for (i = 0; i < R; i++)
        for (j = 0; j < N; j++)
            w[i][j] = frand() - 0.5;

    while (Error > desired_error && q < MAX_EPOCH)
    {
        q++;
        Error = 0;
        for (p = 0; p < 3; p++)
        {
            for (i = 0; i < R; i++)
            {
                double temp = 0;
                for (j = 0; j < N; j++) temp += w[i][j] * ds->x[p][j];
                o[i] = sigmoid(temp);
            }
            for (i = 0; i < R; i++)
                Error += 0.5 * pow(ds->d[p][i] - o[i], 2.0);
            for (i = 0; i < R; i++)
            {
                delta = (ds->d[p][i] - o[i]) * (1 - o[i] * o[i]) / 2;
                for (j = 0; j < N; j++)
                    w[i][j] += eta * delta * ds->x[p][j];
            }
        }
        if (fp) fprintf(fp, "%d,%f\n", q, Error);
    }
    return (Error <= desired_error) ? q : -1;
}

int main()
{
    Dataset datasets[] = {
        {"original (far)",
         {{10, 2, -1}, {2, -5, -1}, {-5, 5, -1}},
         {{1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}},
        {"medium",
         {{3, 1, -1}, {1, -2, -1}, {-2, 2, -1}},
         {{1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}},
        {"close",
         {{1.5, 0.5, -1}, {0.5, -1.0, -1}, {-1.0, 1.0, -1}},
         {{1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}},
        {"very close",
         {{1.0, 0.3, -1}, {0.3, -0.5, -1}, {-0.5, 0.5, -1}},
         {{1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}},
    };
    int n_datasets = 4;
    int n_trials = 10;
    unsigned int base_seed = 42;
    double eta = 0.5;
    int ds_idx, t;
    FILE *fp, *detail_fp;
    char filename[256];

    fp = fopen("close_centers_summary.csv", "w");
    fprintf(fp, "dataset,method,avg_epochs,min_epochs,max_epochs,convergence_rate\n");

    for (ds_idx = 0; ds_idx < n_datasets; ds_idx++)
    {
        int p_total = 0, p_min = MAX_EPOCH + 1, p_max = 0, p_conv = 0;
        int d_total = 0, d_min = MAX_EPOCH + 1, d_max = 0, d_conv = 0;

        for (t = 0; t < n_trials; t++)
        {
            unsigned int seed = base_seed + t;
            int ep;

            ep = run_perceptron(&datasets[ds_idx], eta, seed, NULL);
            if (ep > 0) { p_total += ep; p_conv++; if (ep < p_min) p_min = ep; if (ep > p_max) p_max = ep; }

            ep = run_delta(&datasets[ds_idx], eta, seed, NULL);
            if (ep > 0) { d_total += ep; d_conv++; if (ep < d_min) d_min = ep; if (ep > d_max) d_max = ep; }
        }

        fprintf(fp, "%s,perceptron,%.1f,%d,%d,%.0f%%\n",
                datasets[ds_idx].name,
                p_conv > 0 ? (double)p_total / p_conv : -1.0,
                p_conv > 0 ? p_min : -1, p_conv > 0 ? p_max : -1,
                100.0 * p_conv / n_trials);
        fprintf(fp, "%s,delta,%.1f,%d,%d,%.0f%%\n",
                datasets[ds_idx].name,
                d_conv > 0 ? (double)d_total / d_conv : -1.0,
                d_conv > 0 ? d_min : -1, d_conv > 0 ? d_max : -1,
                100.0 * d_conv / n_trials);

        printf("%-20s  Perceptron: avg=%.1f (conv=%d/%d)  Delta: avg=%.1f (conv=%d/%d)\n",
               datasets[ds_idx].name,
               p_conv > 0 ? (double)p_total / p_conv : -1.0, p_conv, n_trials,
               d_conv > 0 ? (double)d_total / d_conv : -1.0, d_conv, n_trials);

        /* Detailed error curve for delta learning */
        sprintf(filename, "close_delta_%d.csv", ds_idx);
        detail_fp = fopen(filename, "w");
        fprintf(detail_fp, "epoch,error\n");
        run_delta(&datasets[ds_idx], eta, 42, detail_fp);
        fclose(detail_fp);

        sprintf(filename, "close_perceptron_%d.csv", ds_idx);
        detail_fp = fopen(filename, "w");
        fprintf(detail_fp, "epoch,error\n");
        run_perceptron(&datasets[ds_idx], eta, 42, detail_fp);
        fclose(detail_fp);
    }
    fclose(fp);

    printf("\nCSV files generated.\n");
    return 0;
}

/************************************************************************************/
/* Mean subtraction preprocessing for Hopfield network                             */
/* Compares recall accuracy with and without mean pattern subtraction              */
/* Directly addresses the spurious attractor issue observed in Part 1              */
/*                                                                                  */
/* Extended by m5301059 SATO Sho.                                                   */
/************************************************************************************/
#include <stdlib.h>
#include <stdio.h>

#define N     120
#define P     4
#define N_ROW 10
#define N_COL 12

int pat[P][N] = {
    /* "0" */
    {1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1},
    /* "2" */
    {1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    /* "4" */
    {1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1,
     1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
     1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1,
     1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1,
     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1},
    /* "8" */
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
     1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1}
};

double w_orig[N][N];
double w_sub[N][N];
double mean_p[N];
double xi_sub[P][N];
int    v[N];

void compute_mean()
{
    for (int i = 0; i < N; i++) {
        mean_p[i] = 0.0;
        for (int mu = 0; mu < P; mu++)
            mean_p[i] += pat[mu][i];
        mean_p[i] /= (double)P;
    }
}

void store_orig()
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            w_orig[i][j] = 0.0;
            for (int mu = 0; mu < P; mu++)
                w_orig[i][j] += (double)pat[mu][i] * pat[mu][j];
            w_orig[i][j] /= (double)P;
        }
        w_orig[i][i] = 0.0;
    }
}

void store_sub()
{
    for (int mu = 0; mu < P; mu++)
        for (int i = 0; i < N; i++)
            xi_sub[mu][i] = pat[mu][i] - mean_p[i];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            w_sub[i][j] = 0.0;
            for (int mu = 0; mu < P; mu++)
                w_sub[i][j] += xi_sub[mu][i] * xi_sub[mu][j];
            w_sub[i][j] /= (double)P;
        }
        w_sub[i][i] = 0.0;
    }
}

/* Recall pattern m from a noisy start using weight matrix w */
void recall_with(double w[][N], int m, double noise, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < N; i++) {
        double r = (double)(rand() % 10001) / 10000.0;
        v[i] = (r < noise) ? -pat[m][i] : pat[m][i];
    }
    for (int iter = 0; iter < 200; iter++) {
        int changed = 0;
        for (int i = 0; i < N; i++) {
            double net = 0.0;
            for (int j = 0; j < N; j++)
                net += w[i][j] * v[j];
            int vn = (net >= 0) ? 1 : -1;
            if (vn != v[i]) { v[i] = vn; changed++; }
        }
        if (!changed) break;
    }
}

int hamming(int m)
{
    int d = 0;
    for (int i = 0; i < N; i++)
        if (v[i] != pat[m][i]) d++;
    return d;
}

void print_state()
{
    for (int r = 0; r < N_ROW; r++) {
        for (int c = 0; c < N_COL; c++)
            printf("%2c", (v[r * N_COL + c] == -1) ? '*' : ' ');
        printf("\n");
    }
    printf("\n");
}

int main()
{
    compute_mean();
    store_orig();
    store_sub();

    const char *names[] = {"\"0\"", "\"2\"", "\"4\"", "\"8\""};
    double noises[]     = {0.00, 0.10, 0.15};

    for (int ni = 0; ni < 3; ni++) {
        double noise = noises[ni];
        printf("=== Noise: %.0f%% ===\n", noise * 100);
        printf("%-8s %17s %17s\n",
               "Pattern", "No preproc (HD)", "Mean-sub (HD)");
        printf("%-8s %17s %17s\n",
               "-------", "---------------", "-------------");
        for (int m = 0; m < P; m++) {
            unsigned int seed = (unsigned int)(m * 31 + (int)(noise * 100) * 7);
            recall_with(w_orig, m, noise, seed);
            int hd_o = hamming(m);
            recall_with(w_sub,  m, noise, seed);
            int hd_s = hamming(m);
            printf("%-8s %17d %17d\n", names[m], hd_o, hd_s);
        }
        printf("\n");
    }

    /* Visual comparison for pattern "0" and "2" at 10% noise */
    double visual_noise = 0.10;
    int    patterns_to_show[] = {0, 1};
    for (int k = 0; k < 2; k++) {
        int m = patterns_to_show[k];
        unsigned int seed = (unsigned int)(m * 31 + (int)(visual_noise * 100) * 7);
        printf("=== Visual: Pattern %s, Noise=10%% ===\n", names[m]);
        printf("[Without preprocessing]\n");
        recall_with(w_orig, m, visual_noise, seed);
        print_state();
        printf("[With mean subtraction]\n");
        recall_with(w_sub, m, visual_noise, seed);
        print_state();
    }

    return 0;
}

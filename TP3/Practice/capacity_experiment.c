/************************************************************************************/
/* Hopfield network capacity experiment                                             */
/* Measures recall accuracy as number of stored patterns P increases               */
/* Uses random ±1 patterns and 10% noise for recall                                */
/*                                                                                  */
/* Extended by m5301059 SATO Sho.                                                   */
/************************************************************************************/
#include <stdlib.h>
#include <stdio.h>

#define N        120
#define MAX_P    24
#define NOISE    0.10
#define N_TRIALS 20
#define MAX_ITER 200

int    pat[MAX_P][N];
double w[N][N];
int    v[N];

void generate(int P, unsigned int seed)
{
    srand(seed);
    for (int mu = 0; mu < P; mu++)
        for (int i = 0; i < N; i++)
            pat[mu][i] = (rand() % 2 == 0) ? 1 : -1;
}

void store(int P)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            w[i][j] = 0.0;
            for (int mu = 0; mu < P; mu++)
                w[i][j] += (double)pat[mu][i] * pat[mu][j];
            w[i][j] /= (double)P;
        }
        w[i][i] = 0.0;
    }
}

/* Returns pixel accuracy (0.0 to 1.0) after recall from noisy input */
double recall(int m, unsigned int noise_seed)
{
    srand(noise_seed);
    for (int i = 0; i < N; i++) {
        double r = (double)(rand() % 10001) / 10000.0;
        v[i] = (r < NOISE) ? -pat[m][i] : pat[m][i];
    }
    for (int iter = 0; iter < MAX_ITER; iter++) {
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
    int correct = 0;
    for (int i = 0; i < N; i++)
        if (v[i] == pat[m][i]) correct++;
    return (double)correct / N;
}

int main()
{
    int   ps[] = {4, 8, 12, 16, 20, 24};
    int   np   = 6;

    printf("P,avg_accuracy\n");

    for (int pi = 0; pi < np; pi++) {
        int    P     = ps[pi];
        double total = 0.0;
        int    count = 0;

        for (int t = 0; t < N_TRIALS; t++) {
            generate(P, t * 137 + 42);
            store(P);
            for (int mu = 0; mu < P; mu++) {
                total += recall(mu, t * 137 + mu + 1000);
                count++;
            }
        }
        printf("%d,%.4f\n", P, total / count);
    }
    return 0;
}

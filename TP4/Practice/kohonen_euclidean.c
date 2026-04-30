/************************************************************************************/
/* WTA network with Euclidean-distance winner selection (k-means style)             */
/* Extended by m5301059 SATO Sho for comparison with cosine-similarity WTA.        */
/************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define I 4
#define M 3
#define P 150
#define alpha 0.1
#define n_update 50

double w[M][I];
double x[P][I];

int ScanIris()
{
    int MAX_LINE_LEN = 50;
    int MAX_FIELDS   = I;
    FILE *fp = fopen("iris.data", "r");
    char  line[50];
    char *token;
    int   iter = 0, field_count = 0;

    if (fp == NULL) { perror("cannot open iris.data"); return 1; }

    while (fgets(line, sizeof(line), fp) != NULL) {
        field_count = 0;
        line[strcspn(line, "\r\n")] = '\0';
        token = strtok(line, ",");
        while (token != NULL && field_count < MAX_FIELDS) {
            x[iter / I][iter % I] = atof(token);
            token = strtok(NULL, ",");
            iter++;
            field_count++;
        }
    }
    fclose(fp);
    return 0;
}

int main()
{
    int m, m0, i, p, q;
    double dist, dist0;

    ScanIris();

    /* Initialize weights from random training samples */
    srand(42);
    for (m = 0; m < M; m++) {
        int rp = rand() % P;
        for (i = 0; i < I; i++)
            w[m][i] = x[rp][i];
    }

    /* Unsupervised learning with Euclidean-distance winner */
    for (q = 0; q < n_update; q++) {
        for (p = 0; p < P; p++) {
            dist0 = 1e18;
            for (m = 0; m < M; m++) {
                dist = 0;
                for (i = 0; i < I; i++)
                    dist += (w[m][i] - x[p][i]) * (w[m][i] - x[p][i]);
                if (dist < dist0) { dist0 = dist; m0 = m; }
            }
            for (i = 0; i < I; i++)
                w[m0][i] += alpha * (x[p][i] - w[m0][i]);
        }
    }

    /* Classify */
    for (p = 0; p < P; p++) {
        dist0 = 1e18;
        for (m = 0; m < M; m++) {
            dist = 0;
            for (i = 0; i < I; i++)
                dist += (w[m][i] - x[p][i]) * (w[m][i] - x[p][i]);
            if (dist < dist0) { dist0 = dist; m0 = m; }
        }
        printf("Pattern[%d] belongs to %d-th class\n", p, m0);
    }

    fprintf(stderr, "Final prototypes:\n");
    for (m = 0; m < M; m++) {
        fprintf(stderr, "  w%d: ", m);
        for (i = 0; i < I; i++) fprintf(stderr, "%.4f ", w[m][i]);
        fprintf(stderr, "\n");
    }
    return 0;
}

/************************************************************************************/
/* C-program for self-organized learning of Kohonen network                         */
/*                                                                                  */
/* The purpose here is to find the representatives of p                             */
/* clusters in the pattern space. If you can provide the                            */
/* the training samples x, and speicify the number p, you                           */
/* can use this program easily                                                      */
/*                                                                                  */
/*  1) Number of input : I                                                          */
/*  2) Number of neurons: M                                                         */
/*  3) Number of training patterns: P                                               */
/*                                                                                  */
/* This program is produced by Qiangfu Zhao and extended by m5301059 SATO Sho.      */
/* You are free to use it for educational purpose                                   */
/************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define I 4
#define M 3
#define P 150
#define alpha 0.5
#define n_update 20

double w[M][I];
double x[P][I];
double y[M];

/****************************************/
/* Scan the Iris dataset                */
/****************************************/
int ScanIris()
{
    int MAX_LINE_LEN=50;   // 1行の最大長
    int MAX_FIELDS=I;     // 1行の最大フィールド数(正解Labelを除くように調整)
    FILE *fp = fopen("iris.data", "r");   // CSVファイルを開く
    char line[MAX_LINE_LEN];
    char *token;
    int iter = 0, field_count = 0;

    if (fp == NULL) {
        perror("ファイルを開けません");
        return EXIT_FAILURE;
    }

    // 1行ずつ読み込み
    while (fgets(line, sizeof(line), fp) != NULL) {
        field_count = 0;
        // 改行を削除
        line[strcspn(line, "\r\n")] = '\0';

        // strtokでカンマ区切りを分割
        token = strtok(line, ",");
        while (token != NULL && field_count < MAX_FIELDS) {
            x[iter/I][iter%I] = atof(token);
            token = strtok(NULL, ",");
            iter++;
            field_count++;
        }
    }

    /* L2-normalize each input vector to match the cosine-similarity WTA */
    for (int p = 0; p < P; p++) {
        double n = 0;
        for (int j = 0; j < I; j++) n += x[p][j] * x[p][j];
        n = sqrt(n);
        for (int j = 0; j < I; j++) x[p][j] /= n;
    }

    fclose(fp);
    return EXIT_SUCCESS;
}

/*************************************************************/
/* Print out the result of the q-th iteration                */
/*************************************************************/
void PrintResult(int q)
{
    int m, i;

    printf("\n\n");
    printf("Results in the %d-th iteration: \n", q);
    for (m = 0; m < M; m++)
    {
        for (i = 0; i < I; i++)
            printf("%5f ", w[m][i]);
        printf("\n");
    }
    printf("\n\n");
}

/*************************************************************/
/* The main program                                          */
/*************************************************************/
int main()
{
    int m, m0, i, p, q;
    double norm, s, s0;

    /* Scan Iris dataset for learning */
    ScanIris();

    /* Initialization of the connection weights */

    srand(42);
    for (m = 0; m < M; m++)
    {
        int rp = (rand() % P);
        for (i = 0; i < I; i++)
            w[m][i] = x[rp][i];
    }
    PrintResult(0);

    /* Unsupervised learning */

    for (q = 0; q < n_update; q++)
    {
        for (p = 0; p < P; p++)
        {
            s0 = -1e9;
            for (m = 0; m < M; m++)
            {
                s = 0;
                for (i = 0; i < I; i++)
                    s += w[m][i] * x[p][i];
                if (s > s0)
                {
                    s0 = s;
                    m0 = m;
                }
            }

            for (i = 0; i < I; i++)
                w[m0][i] += alpha * (x[p][i] - w[m0][i]);

            norm = 0;
            for (i = 0; i < I; i++)
                norm += w[m0][i] * w[m0][i];
            norm = sqrt(norm);
            for (i = 0; i < I; i++)
                w[m0][i] /= norm;
        }
        PrintResult(q);
    }

    /* Classify the training patterns */

    for (p = 0; p < P; p++)
    {
        s0 = -1e9;
        for (m = 0; m < M; m++)
        {
            s = 0;
            for (i = 0; i < I; i++)
                s += w[m][i] * x[p][i];
            if (s > s0)
            {
                s0 = s;
                m0 = m;
            }
        }
        printf("Pattern[%d] belongs to %d-th class\n", p, m0);
    }
}

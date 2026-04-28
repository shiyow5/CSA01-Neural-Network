/*************************************************************/
/* C-program for BP algorithm (4-bit Parity Check)           */
/*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h> 

#define I             5  /* 4ビットの入力 + 1つのダミー */
#define J             7  /* 隠れ層6個 + 1つのダミー  */
#define K             1  /* 出力層 */
#define n_sample      16 /* 4ビットの全組み合わせ数 */
#define eta           0.5
#define lambda        1.0
#define desired_error 0.001
#define sigmoid(x)    (1.0/(1.0+exp(-lambda*x)))
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))

/* 4ビットの全入出力パターン定義 */
double x[n_sample][I]={
  {0,0,0,0,-1}, // 0
  {0,0,0,1,-1}, // 1
  {0,0,1,0,-1}, // 2
  {0,0,1,1,-1}, // 3
  {0,1,0,0,-1}, // 4
  {0,1,0,1,-1}, // 5
  {0,1,1,0,-1}, // 6
  {0,1,1,1,-1}, // 7
  {1,0,0,0,-1}, // 8
  {1,0,0,1,-1}, // 9
  {1,0,1,0,-1}, // 10
  {1,0,1,1,-1}, // 11
  {1,1,0,0,-1}, // 12
  {1,1,0,1,-1}, // 13
  {1,1,1,0,-1}, // 14
  {1,1,1,1,-1}  // 15
};

/* 1の数が偶数なら1、奇数なら0 */
double d[n_sample][K]={
  {1}, {0}, {0}, {1},  // 0000(偶), 0001(奇), 0010(奇), 0011(偶)
  {0}, {1}, {1}, {0},  // 0100(奇), 0101(偶), 0110(偶), 0111(奇)
  {0}, {1}, {1}, {0},  // 1000(奇), 1001(偶), 1010(偶), 1011(奇)
  {1}, {0}, {0}, {1}   // 1100(偶), 1101(奇), 1110(奇), 1111(偶)
};

double v[J][I],w[K][J];
double y[J];
double o[K];

void Initialization(void);
void FindHidden(int p);
void FindOutput(void);
void PrintResult(void);

int main(){ // 警告が出ないよう int main() に変更
  int    i,j,k,p,q=0;
  double Error=DBL_MAX;
  double delta_o[K];
  double delta_y[J];

  Initialization();
  
  // 無限ループ（局所解）を防止するために、上限を100万回に設定しています
  while(Error>desired_error && q < 1000000){
    q++;
    Error=0;
    for(p=0; p<n_sample; p++){
      FindHidden(p);
      FindOutput();

      for(k=0;k<K;k++){
	      Error += 0.5*pow(d[p][k]-o[k], 2.0);
	      delta_o[k]=(d[p][k]-o[k])*(1-o[k])*o[k];
      }
      
      for(j=0; j<J; j++){
	      delta_y[j]=0;
	      for(k=0;k<K;k++)
	        delta_y[j]+=delta_o[k]*w[k][j];
	      delta_y[j]=(1-y[j])*y[j]*delta_y[j];
      }
	
      for(k=0; k<K; k++)
	      for(j=0; j<J; j++)
	        w[k][j] += eta*delta_o[k]*y[j];
	
      for(j=0; j<J; j++)
	      for(i=0; i<I; i++)
	        v[j][i] += eta*delta_y[j]*x[p][i];
    }
    
    // 表示量が多すぎると処理が遅くなるため、1000回ごとに画面へ出力します
    if (q % 1000 == 0) {
      printf("Error in the %d-th learning cycle = %f\n",q,Error);
    }
  } 
  
  printf("\nFinished at the %d-th learning cycle with Error = %f\n", q, Error);

  PrintResult();
  return 0; // 戻り値を追加
}
  
/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void){
  int i,j,k;

  randomize();
  for(j=0; j<J; j++)
    for(i=0; i<I; i++)
      v[j][i] = frand()-0.5;

  for(k=0; k<K; k++)
    for(j=0; j<J; j++)
      w[k][j] = frand()-0.5;
}

/*************************************************************/
/* Find the output of the hidden neurons                     */
/*************************************************************/
void FindHidden(int p){
  int    i,j;
  double temp;

  for(j=0;j<J-1;j++){
    temp=0;
    for(i=0;i<I;i++)
      temp+=v[j][i]*x[p][i];
    y[j]=sigmoid(temp);
  }
  y[J-1]=-1;
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(void){
  int    j,k;
  double temp;

  for(k=0;k<K;k++){
    temp=0;
    for(j=0;j<J;j++)
      temp += w[k][j]*y[j];
    o[k]=sigmoid(temp);
  }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void){
  int i,j,k;

  printf("\n\n");
  printf("The connection weights in the output layer:\n");
  for(k=0; k<K; k++){
    for(j=0; j<J; j++)
      printf("%5f ",w[k][j]);
    printf("\n");
  }

  printf("\n\n");
  printf("The connection weights in the hidden layer:\n");
  for(j=0; j<J-1; j++){
    for(i=0; i<I; i++)
      printf("%5f ",v[j][i]);
    printf("\n");
  }
  printf("\n\n");
}

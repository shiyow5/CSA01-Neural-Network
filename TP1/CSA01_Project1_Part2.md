# CSA01 - Team Project 1, Part 2: Single Layer Neural Network

**Course:** Neural Networks (CSA01)
**Team Members:**
- SATO Sho (m5301059)
- USAMI Yuki (m5301073)
- SEKINE Kento (m5301060)
- AIZAWA Yuma (m5301001)
- WATABE Chitose (m5301074)

---

## a) Problem Description

In Part 2, we extended the single-neuron programs from Part 1 to a **single-layer neural network** with multiple output neurons. The task is to classify 3 input patterns into 3 classes.

The network has N=3 inputs (2 data inputs + 1 bias of -1) and R=3 output neurons, one for each class. The training data:

| Sample | Input x1 | Input x2 | Bias | Desired Output (o1, o2, o3) |
|:------:|:--------:|:--------:|:----:|:---------------------------:|
| 1      | 10       | 2        | -1   | (+1, -1, -1)               |
| 2      | 2        | -5       | -1   | (-1, +1, -1)               |
| 3      | -5       | 5        | -1   | (-1, -1, +1)               |

The correct class gets +1, and the other two get -1.

### Source Code: Perceptron Learning Rule (`perceptron_learning_NN.c`)

```c
/*************************************************************/
/* C-program for learning of single layer neural network     */
/* based on the delta learning rule                          */
/*                                                           */
/*  1) Number of Inputs : N                                  */
/*  2) Number of Output : R                                  */
/* The last input for all neurons is always -1               */
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
#define eta 0.5
#define lambda 1.0
#define desired_error 0.1
#define stepf(x) (x >= 0 ? 1 : -1)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

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
double w[R][N];
double o[R];

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main()
{
  int i, j, p, q = 0;
  double Error = DBL_MAX;
  double delta, LearningSignal;

  Initialization();
  while (Error > desired_error)
  {
    q++;
    Error = 0;
    for (p = 0; p < n_sample; p++)
    {
      FindOutput(p);
      for (i = 0; i < R; i++)
      {
        Error += 0.5 * pow(d[p][i] - o[i], 2.0);
      }
      for (i = 0; i < R; i++)
      {
        LearningSignal = eta * (d[p][i] - o[i]);
        for (j = 0; j < N; j++)
        {
          w[i][j] += LearningSignal * x[p][j];
        }
      }
    }
    printf("Error in the %d-th learning cycle=%f\n", q, Error);
  }
  PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
  int i, j;

  randomize();
  for (i = 0; i < R; i++)
    for (j = 0; j < N; j++)
      w[i][j] = frand() - 0.5;
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p)
{
  int i, j;
  double temp;

  for (i = 0; i < R; i++)
  {
    temp = 0;
    for (j = 0; j < N; j++)
    {
      temp += w[i][j] * x[p][j];
    }
    o[i] = stepf(temp);
  }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void)
{
  int i, j, p;

  printf("\n\n");
  printf("The connection weights are:\n");
  for (i = 0; i < R; i++)
  {
    for (j = 0; j < N; j++)
      printf("%5f ", w[i][j]);
    printf("\n");
  }
  printf("\n\n");

  printf("Neuron output for each input pattern:\n");
  for (p = 0; p < n_sample; p++)
  {
    FindOutput(p);
    printf("(");
    for (i = 0; i < N; i++)
      printf(" %.1f,", x[p][i]);
    printf(") -> (");
    for (i = 0; i < R; i++)
      printf(" %5f,", o[i]);
    printf(")\n");
  }
  printf("\n");
}
```

### Source Code: Delta Learning Rule (`delta_learning_NN.c`)

```c
/***********************************************************************************/
/* C-program for learning of single layer neural network                           */
/* based on the delta learning rule                                                */
/*                                                                                 */
/*  1) Number of Inputs : N                                                        */
/*  2) Number of Output : R                                                        */
/* The last input for all neurons is always -1                                     */
/*                                                                                 */
/* This program is produced by Qiangfu Zhao and extended by m5301059 SATO Sho.     */
/* You are free to use it for educational purpose                                  */
/***********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 3
#define R 3
#define n_sample 3
#define eta 0.5
#define lambda 1.0
#define desired_error 0.1
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

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
double w[R][N];
double o[R];

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main()
{
  int i, j, p, q = 0;
  double Error = DBL_MAX;
  double delta;

  Initialization();
  while (Error > desired_error)
  {
    q++;
    Error = 0;
    for (p = 0; p < n_sample; p++)
    {
      FindOutput(p);
      for (i = 0; i < R; i++)
      {
        Error += 0.5 * pow(d[p][i] - o[i], 2.0);
      }
      for (i = 0; i < R; i++)
      {
        delta = (d[p][i] - o[i]) * (1 - o[i] * o[i]) / 2;
        for (j = 0; j < N; j++)
        {
          w[i][j] += eta * delta * x[p][j];
        }
      }
    }
    printf("Error in the %d-th learning cycle=%f\n", q, Error);
  }
  PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
  int i, j;

  randomize();
  for (i = 0; i < R; i++)
    for (j = 0; j < N; j++)
      w[i][j] = frand() - 0.5;
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p)
{
  int i, j;
  double temp;

  for (i = 0; i < R; i++)
  {
    temp = 0;
    for (j = 0; j < N; j++)
    {
      temp += w[i][j] * x[p][j];
    }
    o[i] = sigmoid(temp);
  }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void)
{
  int i, j, p;

  printf("\n\n");
  printf("The connection weights are:\n");
  for (i = 0; i < R; i++)
  {
    for (j = 0; j < N; j++)
      printf("%5f ", w[i][j]);
    printf("\n");
  }
  printf("\n\n");

  printf("Neuron output for each input pattern:\n");
  for (p = 0; p < n_sample; p++)
  {
    FindOutput(p);
    printf("(");
    for (i = 0; i < N; i++)
      printf(" %.1f,", x[p][i]);
    printf(") -> (");
    for (i = 0; i < R; i++)
      printf(" %5f,", o[i]);
    printf(")\n");
  }
  printf("\n");
}
```

## b) Methods Used

### Perceptron Learning Rule (Single Layer Network)

Each of the 3 output neurons has its own weight vector of length 3. The activation is a step function (same as Part 1). The weight update for output neuron i is:

```
w_ij(t+1) = w_ij(t) + eta * (d_i - o_i) * x_j
```

The total error sums the squared errors over all output neurons and all samples.

### Delta Learning Rule (Single Layer Network)

Same sigmoid activation as Part 1, applied independently to each output neuron:

```
delta_i = (d_i - o_i) * (1 - o_i^2) / 2
w_ij(t+1) = w_ij(t) + eta * delta_i * x_j
```

### Common Setup

- Learning rate (eta): 0.5
- Convergence threshold: 0.1 (looser than Part 1's 0.01)
- Weights initialized randomly in [-0.5, 0.5)

## c) Discussion of Simulation Results

### Perceptron Learning Results

Converged in **4 epochs**. Final weight matrix:

```
Neuron 1: [4.121,  4.974,  4.615]
Neuron 2: [-2.803, -12.022, 1.111]
Neuron 3: [-9.803, -1.609,  0.993]
```

| Input           | Output              | Expected            |
|:---------------:|:-------------------:|:-------------------:|
| (10, 2, -1)     | (+1, -1, -1)        | (+1, -1, -1)        |
| (2, -5, -1)     | (-1, +1, -1)        | (-1, +1, -1)        |
| (-5, 5, -1)     | (-1, -1, +1)        | (-1, -1, +1)        |

All correct.

### Delta Learning Results

Converged in **3 epochs**. Final weight matrix:

```
Neuron 1: [2.224,  1.556, -0.004]
Neuron 2: [-1.323, -2.837, 0.455]
Neuron 3: [-0.485,  0.392, 0.249]
```

| Input           | Output                      | Expected            |
|:---------------:|:---------------------------:|:-------------------:|
| (10, 2, -1)     | (+1.000, -1.000, -0.974)    | (+1, -1, -1)        |
| (2, -5, -1)     | (-0.931, +1.000, -0.920)    | (-1, +1, -1)        |
| (-5, 5, -1)     | (-0.931, -0.999, +0.968)    | (-1, -1, +1)        |

All outputs are close to the targets. The correct class always has the highest output.

### Comparison

Interestingly, the delta rule (3 epochs) converged faster than the perceptron (4 epochs) here. In Part 1 it was the opposite (5 vs. 969). The likely reason is that the convergence threshold in Part 2 is 0.1 instead of 0.01, and the input values are much larger (10, -5, etc. vs. binary 0/1). With big inputs, even a small weight change causes a big change in the weighted sum, so the sigmoid can reach close-to-target values quickly.

The perceptron produced larger weight magnitudes (up to 12) compared to the delta rule (up to about 3). This is because the perceptron's error signal is always 0 or +/-2 (discrete), leading to big jumps, while the delta rule's gradient-based updates are more moderate.

Both methods correctly classified all patterns. The key difference is that the perceptron gives exact +1/-1 outputs while the delta rule gives approximations like -0.974. For classification purposes, both work fine on this problem, since the class with the highest output value is always the correct one.

---

## d) New Problem: Closer Class Centers

We wanted to see how the distance between class centers affects learning. The original data has well-separated centers (10,2), (2,-5), (-5,5), so we created 3 additional datasets with progressively closer centers:

| Dataset      | Class 1    | Class 2     | Class 3     |
|:------------:|:----------:|:-----------:|:-----------:|
| Original     | (10, 2)    | (2, -5)     | (-5, 5)     |
| Medium       | (3, 1)     | (1, -2)     | (-2, 2)     |
| Close        | (1.5, 0.5) | (0.5, -1.0) | (-1.0, 1.0) |
| Very close   | (1.0, 0.3) | (0.3, -0.5) | (-0.5, 0.5) |

We ran 10 trials with different random seeds for each dataset (eta = 0.5, threshold = 0.1).

| Dataset      | Perceptron Avg Epochs | Delta Avg Epochs |
|:------------:|:---------------------:|:----------------:|
| Original     | 3.2                   | 11.9             |
| Medium       | 3.0                   | 9.8              |
| Close        | 2.8                   | 27.6             |
| Very close   | 3.4                   | 79.7             |

All trials converged.

The perceptron barely changed — about 3 epochs regardless of distance. Since the step function only cares about the sign of the weighted sum (is it positive or negative?), the magnitude of the inputs does not matter much.

The delta rule, however, got much slower as the centers got closer. Going from "original" to "very close", the average went from 12 to 80 epochs. When the inputs are similar, the sigmoid outputs are also similar for different patterns, so the error signals become smaller and learning slows down.

![Close Centers Comparison](Practice/Part2/close_centers.png)

*Figure 3: Effect of class distance on convergence. Bottom-right shows the input distributions for all four datasets.*

---

## e) Modification: Weight Initialization Influence

We tested what happens when you change the range of the initial random weights. Using the original dataset, we tried 6 ranges: [-0.01, 0.01], [-0.1, 0.1], [-0.5, 0.5], [-1.0, 1.0], [-2.0, 2.0], [-5.0, 5.0]. Each was run 20 times with different seeds.

| Init Range | Delta Avg Epochs | Delta Std Dev | Delta Conv. Rate | Perceptron Avg | Perceptron Conv. |
|:----------:|:----------------:|:-------------:|:----------------:|:--------------:|:----------------:|
| 0.01       | 4.2              | 0.8           | 100%             | 3.2            | 100%             |
| 0.10       | 4.0              | 1.4           | 100%             | 3.2            | 100%             |
| 0.50       | 9.5              | 9.0           | 100%             | 3.2            | 100%             |
| 1.00       | 87.5             | 172.4         | 95%              | 3.3            | 100%             |
| 2.00       | 4521.1           | 9465.3        | 80%              | 2.6            | 100%             |
| 5.00       | 7579.2           | 7051.6        | 20%              | 2.8            | 100%             |

The perceptron was completely unaffected — it converged in about 3 epochs no matter what the initial weights were.

The delta rule, on the other hand, was very sensitive. With small initial weights (range 0.01~0.10), it converged fast and consistently (about 4 epochs). But at range 5.0, only 4 out of 20 trials converged within 50,000 epochs, and even the ones that did took thousands of epochs on average.

The reason is that large initial weights push the sigmoid into saturation (output close to +1 or -1), where the derivative `(1-o^2)/2` is nearly zero. When the gradient is that small, the weight updates are tiny and learning basically stalls. This is essentially the "vanishing gradient" problem. The standard deviation also blew up at larger ranges (from 0.8 to 9,465), meaning the results become very unpredictable.

![Weight Initialization Comparison](Practice/Part2/weight_init.png)

*Figure 4: Effect of weight initialization range. Bottom-right shows how the convergence rate drops sharply for large ranges.*

---

## f) Discussion on Extended Experiments

From the class distance experiment (d), we found that the perceptron is not really affected by how close the classes are, while the delta rule gets noticeably slower. This makes sense because the step function does not care about magnitudes, but the sigmoid does. For harder problems where classes overlap more, the delta rule would need more epochs or a better learning rate.

From the weight initialization experiment (e), we saw that starting with large weights is bad for the delta rule because of sigmoid saturation. Small initial weights (close to zero) keep the sigmoid in its linear region where the gradient is largest, so learning is fast and reliable. The perceptron does not have this problem since the step function is not affected by the magnitude of the weighted sum.

Overall, the delta rule is more powerful (differentiable, extendable to multi-layer networks) but also more fragile — you have to be careful with the learning rate and weight initialization. The perceptron is simpler and more robust for these kinds of problems, but it cannot be extended to multi-layer learning.

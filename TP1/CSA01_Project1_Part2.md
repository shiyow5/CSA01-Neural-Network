# CSA01 - Team Project 1, Part 2: Single Layer Neural Network

**Course:** Neural Networks (CSA01)
**Team Members:** SATO Sho (m5301059)

---

## a) Problem Description

The objective of Part 2 is to extend the single-neuron learning from Part 1 to a **single-layer neural network** with multiple output neurons. The task is a **3-class classification problem** where the network must learn to classify 3 input patterns into 3 distinct classes.

The network architecture is:
- **Input neurons (N):** 3 (2 data inputs + 1 bias input of -1)
- **Output neurons (R):** 3 (one per class)

The training data consists of 3 samples:

| Sample | Input x1 | Input x2 | Bias | Desired Output (o1, o2, o3) |
|:------:|:--------:|:--------:|:----:|:---------------------------:|
| 1      | 10       | 2        | -1   | (+1, -1, -1)               |
| 2      | 2        | -5       | -1   | (-1, +1, -1)               |
| 3      | -5       | 5        | -1   | (-1, -1, +1)               |

Each output neuron represents one class. The desired output is +1 for the correct class and -1 for all other classes (one-hot encoding in {-1, +1}).

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

Each of the R=3 output neurons has its own set of N=3 weights. The step function is used as the activation function:

```
f(x) = +1  if x >= 0
f(x) = -1  if x < 0
```

For each output neuron i, the weight update rule is:

```
w_ij(t+1) = w_ij(t) + eta * (d_i - o_i) * x_j
```

where `w_ij` is the weight from input j to output neuron i. The total error is the sum of squared errors across all output neurons and all samples.

### Delta Learning Rule (Single Layer Network)

The delta learning rule extends to the multi-output case using the sigmoid activation function:

```
f(x) = 2 / (1 + exp(-lambda * x)) - 1,  where lambda = 1.0
```

For each output neuron i, the weight update is:

```
delta_i = (d_i - o_i) * (1 - o_i^2) / 2
w_ij(t+1) = w_ij(t) + eta * delta_i * x_j
```

Each output neuron independently learns its own decision boundary through its dedicated weight vector.

### Common Setup

- Learning rate (eta): 0.5
- Convergence threshold: 0.1
- Weights initialized randomly in [-0.5, 0.5)

## c) Discussion of Simulation Results

### Perceptron Learning Results

The perceptron learning rule converged in **4 epochs**. The final weight matrix was:

```
Neuron 1: [4.121,  4.974,  4.615]
Neuron 2: [-2.803, -12.022, 1.111]
Neuron 3: [-9.803, -1.609,  0.993]
```

The final outputs for all input patterns were:

| Input           | Output              | Expected            |
|:---------------:|:-------------------:|:-------------------:|
| (10, 2, -1)     | (+1, -1, -1)        | (+1, -1, -1)        |
| (2, -5, -1)     | (-1, +1, -1)        | (-1, +1, -1)        |
| (-5, 5, -1)     | (-1, -1, +1)        | (-1, -1, +1)        |

All patterns were classified perfectly with exact +1/-1 outputs.

### Delta Learning Results

The delta learning rule converged in **3 epochs**. The final weight matrix was:

```
Neuron 1: [2.224,  1.556, -0.004]
Neuron 2: [-1.323, -2.837, 0.455]
Neuron 3: [-0.485,  0.392, 0.249]
```

The final outputs for all input patterns were:

| Input           | Output                      | Expected            |
|:---------------:|:---------------------------:|:-------------------:|
| (10, 2, -1)     | (+1.000, -1.000, -0.974)    | (+1, -1, -1)        |
| (2, -5, -1)     | (-0.931, +1.000, -0.920)    | (-1, +1, -1)        |
| (-5, 5, -1)     | (-0.931, -0.999, +0.968)    | (-1, -1, +1)        |

All outputs are close to the desired values. The correct class neuron consistently has the highest output value.

### Comparison and Discussion

1. **Convergence Speed:** In this case, the delta learning rule converged slightly faster (3 epochs) than the perceptron (4 epochs). This contrasts with Part 1 where the perceptron was much faster. The reason is that the convergence threshold is more relaxed (0.1 vs. 0.01), and the sigmoid function can gradually reduce the error across all neurons simultaneously, while the perceptron's discrete outputs may cause oscillation in the error across different patterns.

2. **Classification Accuracy:** Both methods correctly classify all three input patterns. The perceptron outputs are exact (+1/-1), while the delta learning outputs are approximate but close to the targets (e.g., -0.974 instead of -1).

3. **Weight Structure:** The perceptron produces larger weight magnitudes (up to 12) compared to the delta learning rule (up to about 3). The perceptron needs larger weights because it adjusts weights by the full error signal `(d-o)`, which is either 0 or +/-2. The delta rule uses a scaled gradient, producing more moderate weight updates.

4. **Input Scale Effect:** The input data in Part 2 has much larger magnitudes (values like 10, -5) compared to Part 1 (binary 0/1). This causes larger weight updates per iteration, leading to faster convergence for both methods. The large separation between class centroids also makes the classification problem relatively easy.

5. **Network Scalability:** Part 2 demonstrates that both learning rules naturally extend from single-neuron to multi-neuron networks. Each output neuron independently learns its own set of weights, effectively solving R separate binary classification problems in parallel.

6. **Limitations of Single-Layer Networks:** While both methods work well for this linearly separable 3-class problem, a single-layer network cannot solve non-linearly separable problems (e.g., XOR). Multi-layer networks with hidden layers and backpropagation would be needed for such problems.

7. **Practical Consideration:** The delta learning rule, despite producing approximate outputs, provides a smoother optimization landscape due to the differentiable sigmoid activation. This property becomes crucial when extending to multi-layer architectures where gradient information must propagate through multiple layers.

---

## d) New Problem: Closer Class Centers

### Problem Description

We investigated how the distance between class centers affects the convergence behavior of both learning rules. Four datasets with progressively closer class centers were tested:

| Dataset      | Class 1 Center | Class 2 Center | Class 3 Center |
|:------------:|:--------------:|:--------------:|:--------------:|
| Original (far) | (10, 2)      | (2, -5)        | (-5, 5)        |
| Medium       | (3, 1)         | (1, -2)        | (-2, 2)        |
| Close        | (1.5, 0.5)     | (0.5, -1.0)    | (-1.0, 1.0)    |
| Very close   | (1.0, 0.3)     | (0.3, -0.5)    | (-0.5, 0.5)    |

All datasets use the same desired outputs (one-hot encoding) and the same learning parameters (eta = 0.5, threshold = 0.1). Each experiment was run for 10 trials with different random seeds.

### Results

| Dataset        | Perceptron Avg Epochs | Delta Avg Epochs |
|:--------------:|:---------------------:|:----------------:|
| Original (far) | 3.2                   | 11.9             |
| Medium         | 3.0                   | 9.8              |
| Close          | 2.8                   | 27.6             |
| Very close     | 3.4                   | 79.7             |

All trials converged (100% convergence rate).

### Analysis

1. **Perceptron Robustness:** The perceptron learning rule is remarkably insensitive to class center distance. Convergence takes 3-4 epochs regardless of how close the classes are. This is because the step function only cares about which side of the decision boundary a point falls on, not how far it is from the boundary.

2. **Delta Learning Sensitivity:** The delta rule's convergence time increases dramatically as class centers get closer:
   - Original (far): 11.9 epochs
   - Very close: 79.7 epochs (6.7x slower)
   
   When classes are close, the sigmoid outputs for different patterns are more similar, producing smaller error signals and smaller weight updates. The network needs many more iterations to develop weights that can discriminate between the similar inputs.

3. **Practical Implication:** In real-world classification tasks, poorly separated classes are common. The delta learning rule's sensitivity to class separation distance suggests that proper data preprocessing (e.g., feature scaling, dimensionality reduction) is important for efficient training.

![Close Centers Comparison](Practice/Part2/close_centers.png)

*Figure 3: Effect of class center distance on convergence. Top: error curves. Bottom-left: average convergence epochs. Bottom-right: input pattern distributions showing the four datasets.*

---

## e) Modification: Weight Initialization Influence

### Description

We studied the effect of weight initialization range on convergence behavior. The original problem (far class centers) was used with six different initialization ranges: **[-0.01, 0.01], [-0.1, 0.1], [-0.5, 0.5], [-1.0, 1.0], [-2.0, 2.0], and [-5.0, 5.0]**. For each range, 20 trials with different random seeds were run.

### Results

| Init Range | Delta Avg Epochs | Delta Std Dev | Delta Conv. Rate | Perceptron Avg Epochs | Perceptron Conv. Rate |
|:----------:|:----------------:|:-------------:|:----------------:|:---------------------:|:---------------------:|
| 0.01       | 4.2              | 0.8           | 100%             | 3.2                   | 100%                  |
| 0.10       | 4.0              | 1.4           | 100%             | 3.2                   | 100%                  |
| 0.50       | 9.5              | 9.0           | 100%             | 3.2                   | 100%                  |
| 1.00       | 87.5             | 172.4         | 95%              | 3.3                   | 100%                  |
| 2.00       | 4521.1           | 9465.3        | 80%              | 2.6                   | 100%                  |
| 5.00       | 7579.2           | 7051.6        | 20%              | 2.8                   | 100%                  |

### Analysis

1. **Perceptron Immunity:** The perceptron is completely unaffected by the initialization range. Convergence takes 2-3 epochs regardless of whether weights start near 0 or at +/-5. This is because the step function output is determined solely by the sign of the weighted sum, not its magnitude.

2. **Delta Learning — Critical Sensitivity:** The delta rule is extremely sensitive to initialization range:
   - **Small range (0.01-0.10):** Fast, consistent convergence (~4 epochs, low variance).
   - **Medium range (0.50):** Still converges but with higher variance (std = 9.0).
   - **Large range (1.00+):** Convergence becomes unreliable. At range = 5.0, only 20% of trials converge within 50,000 epochs.

3. **Saturation Problem:** Large initial weights cause the sigmoid function to saturate (output near +1 or -1), where the derivative `(1-o^2)/2` approaches zero. This creates a "vanishing gradient" that prevents effective learning. This is a well-known problem in neural network training and motivates techniques like Xavier/He initialization in modern deep learning.

4. **Variance Explosion:** As initialization range increases, the standard deviation of convergence time explodes (from 0.8 at range=0.01 to 9,465 at range=2.0), indicating highly unpredictable training behavior with large initial weights.

![Weight Initialization Comparison](Practice/Part2/weight_init.png)

*Figure 4: Effect of weight initialization on learning. Top: error curves for different init ranges. Bottom-left: convergence epochs with error bars (log scale). Bottom-right: convergence rate drops sharply for large initialization ranges.*

---

## f) Discussion on Extended Experiments

### Summary of Findings

The extended experiments in d) and e) provide practical insights into single-layer neural network training:

1. **Class Separation and Convergence:** The closer the class centers, the harder it is for the delta learning rule to converge. The perceptron remains robust, but only because it makes binary decisions. In practice, the delta rule's sensitivity motivates the use of feature engineering and data normalization to maximize class separation before training.

2. **Weight Initialization is Critical:** For gradient-based learning (delta rule), proper weight initialization is essential. Small initial weights (near zero) ensure the neurons operate in the linear region of the sigmoid, where gradients are largest. Large initial weights push neurons into saturation, causing vanishing gradients and training failure. This finding directly connects to modern initialization techniques (Xavier, He) that are standard practice in deep learning.

3. **Perceptron vs. Delta Trade-off:** The perceptron learning rule is more robust to hyperparameter choices and data characteristics for simple, linearly separable problems. However, its non-differentiable step function prevents it from being used in multi-layer networks. The delta rule, despite being more sensitive to hyperparameters, provides the gradient information necessary for training deeper architectures.

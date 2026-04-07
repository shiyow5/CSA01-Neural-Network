# CSA01 - Team Project 1, Part 1: Single Neuron Learning

**Course:** Neural Networks (CSA01)
**Team Members:** SATO Sho (m5301059)

---

## a) Problem Description

The objective of Part 1 is to implement and compare two learning rules for a single artificial neuron: the **Perceptron Learning Rule** and the **Delta Learning Rule**.

We chose the **AND gate** as the target problem. The AND gate is a linearly separable binary function, making it suitable for single-neuron learning. The training data consists of 4 samples with 2-dimensional binary inputs (plus a bias input of -1):

| Input x1 | Input x2 | Bias | Desired Output d |
|:---------:|:---------:|:----:|:----------------:|
| 0         | 0         | -1   | -1               |
| 0         | 1         | -1   | -1               |
| 1         | 0         | -1   | -1               |
| 1         | 1         | -1   | 1                |

The output is +1 only when both inputs are 1 (AND logic), and -1 otherwise.

### Source Code: Perceptron Learning Rule (`perceptron_learning.c`)

```c
/*************************************************************/
/* C-program for perceptron-learning rule                    */
/* Learning rule of one neuron                               */
/*                                                           */
/* This program is produced by m5301059 SATO Sho.            */
/*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I 3
#define n_sample 4
#define eta 0.5
#define lambda 1.0
#define desired_error 0.01
#define stepf(x) (x >= 0 ? 1 : -1)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

// 2d { x, y, dummy input for bias(=-1)}
double x[n_sample][I] = {
    {0, 0, -1},
    {0, 1, -1},
    {1, 0, -1},
    {1, 1, -1},
};

double w[I];
double d[n_sample] = {-1, -1, -1, 1};
double o;

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main()
{
  int i, p, q = 0;
  double delta, Error = DBL_MAX, LearningSignal;

  Initialization();
  while (Error > desired_error)
  {
    q++;
    Error = 0;
    for (p = 0; p < n_sample; p++)
    {
      FindOutput(p);
      Error += 0.5 * pow(d[p] - o, 2.0);
      LearningSignal = eta * (d[p] - o);
      for (i = 0; i < I; i++)
      {
        w[i] += LearningSignal * x[p][i];
      }
      printf("Error in the %d-th learning cycle=%f\n", q, Error);
    }
  }
  PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
  int i;

  randomize();
  for (i = 0; i < I; i++)
    w[i] = frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p)
{
  int i;
  double temp = 0;

  for (i = 0; i < I; i++)
    temp += w[i] * x[p][i];
  o = stepf(temp);
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void)
{
  int i, p;

  printf("\n\n");
  printf("The connection weights of the neurons:\n");
  for (i = 0; i < I; i++)
    printf("%5f ", w[i]);
  printf("\n\n");

  printf("Neuron output for each input pattern:\n");
  for (p = 0; p < n_sample; p++)
  {
    FindOutput(p);
    printf("(");
    for (i = 0; i < I; i++)
      printf(" %.1f,", x[p][i]);
    printf(") -> %5f\n", o);
  }
  printf("\n");
}
```

### Source Code: Delta Learning Rule (`delta_learning.c`)

```c
/*********************************************************************************/
/* C-program for delta-learning rule                                             */
/* Learning rule of one neuron                                                   */
/*                                                                               */
/* This program is produced by Qiangfu Zhao and extended by m5301059 SATO Sho.   */
/* You are free to use it for educational purpose                                */
/*********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I 3
#define n_sample 4
#define eta 0.5
#define lambda 1.0
#define desired_error 0.01
#define sigmoid(x) (2.0 / (1.0 + exp(-lambda * x)) - 1.0)
#define frand() (rand() % 10000 / 10001.0)
#define randomize() srand((unsigned int)time(NULL))

// 2d { x, y, dummy input for bias(=-1)}
double x[n_sample][I] = {
    {0, 0, -1},
    {0, 1, -1},
    {1, 0, -1},
    {1, 1, -1},
};

double w[I];
double d[n_sample] = {-1, -1, -1, 1};
double o;

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main()
{
    int i, p, q = 0;
    double delta, Error = DBL_MAX;

    Initialization();
    while (Error > desired_error)
    {
        q++;
        Error = 0;
        for (p = 0; p < n_sample; p++)
        {
            FindOutput(p);
            Error += 0.5 * pow(d[p] - o, 2.0);
            for (i = 0; i < I; i++)
            {
                delta = (d[p] - o) * (1 - o * o) / 2;
                w[i] += eta * delta * x[p][i];
            }
            printf("Error in the %d-th learning cycle=%f\n", q, Error);
        }
    }
    PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void)
{
    int i;

    randomize();
    for (i = 0; i < I; i++)
        w[i] = frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p)
{
    int i;
    double temp = 0;

    for (i = 0; i < I; i++)
        temp += w[i] * x[p][i];
    o = sigmoid(temp);
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void)
{
    int i, p;

    printf("\n\n");
    printf("The connection weights of the neurons:\n");
    for (i = 0; i < I; i++)
        printf("%5f ", w[i]);
    printf("\n\n");

    printf("Neuron output for each input pattern:\n");
    for (p = 0; p < n_sample; p++)
    {
        FindOutput(p);
        printf("(");
        for (i = 0; i < I; i++)
            printf(" %.1f,", x[p][i]);
        printf(") -> %5f\n", o);
    }
    printf("\n");
}
```

## b) Methods Used

### Perceptron Learning Rule

The perceptron learning rule uses a **step (sign) function** as the activation function:

```
f(x) = +1  if x >= 0
f(x) = -1  if x < 0
```

The weight update rule is:

```
w_i(t+1) = w_i(t) + eta * (d - o) * x_i
```

where `eta = 0.5` is the learning rate, `d` is the desired output, `o` is the actual output, and `x_i` is the i-th input. The error for each pattern is computed as `E = 0.5 * (d - o)^2`. Learning stops when the total error across all samples falls below 0.01.

### Delta Learning Rule

The delta learning rule uses a **sigmoid activation function** scaled to the range [-1, 1]:

```
f(x) = 2 / (1 + exp(-lambda * x)) - 1,  where lambda = 1.0
```

The weight update rule is:

```
delta = (d - o) * (1 - o^2) / 2
w_i(t+1) = w_i(t) + eta * delta * x_i
```

The term `(1 - o^2) / 2` is the derivative of the sigmoid function. This allows gradient-based continuous weight adjustment. The same convergence criterion (total error < 0.01) is used.

### Common Setup

- Number of inputs: 3 (2 data inputs + 1 bias)
- Learning rate (eta): 0.5
- Weights are initialized randomly in [0, 1)
- Convergence threshold: 0.01

## c) Discussion of Simulation Results

### Perceptron Learning Results

The perceptron learning rule converged in **5 epochs**. The final connection weights were:

```
w = [1.528, 1.227, 2.427]
```

The final outputs for all input patterns were:

| Input       | Output | Expected |
|:-----------:|:------:|:--------:|
| (0, 0, -1)  | -1     | -1       |
| (0, 1, -1)  | -1     | -1       |
| (1, 0, -1)  | -1     | -1       |
| (1, 1, -1)  | +1     | +1       |

All outputs match the desired values exactly. The perceptron perfectly classified the AND function.

### Delta Learning Results

The delta learning rule converged in **969 epochs**. The final connection weights were:

```
w = [6.281, 6.278, 9.503]
```

The final outputs for all input patterns were:

| Input       | Output    | Expected |
|:-----------:|:---------:|:--------:|
| (0, 0, -1)  | -0.9999   | -1       |
| (0, 1, -1)  | -0.9235   | -1       |
| (1, 0, -1)  | -0.9233   | -1       |
| (1, 1, -1)  | +0.9101   | +1       |

All outputs are close to the desired values but not exact, as the sigmoid function is continuous and asymptotically approaches the target values.

### Comparison and Discussion

1. **Convergence Speed:** The perceptron learning rule converged significantly faster (5 epochs) compared to the delta learning rule (969 epochs). This is because the step function produces exact +1/-1 outputs, so the error drops to zero as soon as the decision boundary is correctly placed. In contrast, the sigmoid function produces continuous outputs that approach but never exactly reach +/-1, requiring many more iterations to reduce the error below the threshold.

2. **Output Precision:** The perceptron outputs are exactly +1 or -1, while the delta learning outputs are approximate values (e.g., +0.91 instead of +1). This is an inherent characteristic of the sigmoid activation function.

3. **Weight Magnitudes:** The delta learning rule produces larger weights (around 6-9) compared to the perceptron (around 1-2). Larger weights are needed for the sigmoid function to produce outputs closer to the saturation values of +/-1.

4. **Decision Boundary:** Both methods learn a valid decision boundary that separates the AND function correctly. The perceptron finds the boundary with minimal weight adjustment, while the delta rule requires the weights to grow larger to push the sigmoid outputs toward saturation.

5. **Differentiability:** The key advantage of the delta learning rule is that it uses a differentiable activation function, which enables gradient-based learning. While this is not necessary for a simple linearly separable problem like AND, it becomes essential for more complex problems and multi-layer networks where backpropagation is required.

6. **Error Trajectory:** The perceptron error decreases in discrete jumps (multiples of 2, since `(d-o)^2 = 4` when d and o differ by 2), while the delta learning error decreases smoothly and gradually, reflecting the continuous nature of the sigmoid activation.

---

## d) New Problem: XOR Gate

### Problem Description

We applied both learning rules to the **XOR (exclusive OR)** problem to demonstrate the fundamental limitation of a single neuron. XOR is a non-linearly separable function:

| Input x1 | Input x2 | Desired Output d |
|:---------:|:---------:|:----------------:|
| 0         | 0         | -1               |
| 0         | 1         | +1               |
| 1         | 0         | +1               |
| 1         | 1         | -1               |

No single straight line can separate the +1 outputs from the -1 outputs in the input space, which means no single neuron can solve this problem regardless of the learning rule used.

### Results

Both learning rules were run for a maximum of 1,000 epochs:

- **Perceptron:** Did NOT converge. Final error = 8.0 (remained constant after initial epochs).
- **Delta Learning:** Did NOT converge. Final error = 2.59 (stabilized around this value).

The perceptron's error stabilized at 8.0 because two out of four patterns are always misclassified (each contributing an error of 4.0). The delta learning rule's error stabilized around 2.59, as the sigmoid outputs settled near 0 (neither +1 nor -1), representing the neuron's inability to find a valid separation.

### Theoretical Explanation

A single neuron computes a linear decision boundary: `w1*x1 + w2*x2 - w_bias = 0`. This is a straight line in 2D space. The XOR function requires two separate regions for the +1 class (corners (0,1) and (1,0)), which cannot be separated from the -1 class (corners (0,0) and (1,1)) by a single line. This is a classic result from Minsky and Papert (1969), demonstrating that perceptrons cannot solve non-linearly separable problems.

![XOR Error Curves](Practice/Part1/xor_error.png)

*Figure 1: Error curves for both learning rules on XOR. Neither converges to the desired error threshold.*

---

## e) Modification: Learning Rate Comparison

### Description

We investigated the effect of the learning rate (eta) on convergence speed and stability for the AND gate problem. Six learning rates were tested: **0.01, 0.1, 0.5, 1.0, 2.0, and 5.0**. For each learning rate, 10 trials with different random seeds were run.

### Results

| eta  | Perceptron Avg Epochs | Perceptron Conv. Rate | Delta Avg Epochs | Delta Conv. Rate |
|:----:|:---------------------:|:---------------------:|:----------------:|:----------------:|
| 0.01 | 9.5                   | 100%                  | N/A              | 0%               |
| 0.10 | 3.5                   | 100%                  | 4947.5           | 100%             |
| 0.50 | 6.1                   | 100%                  | 971.6            | 100%             |
| 1.00 | 6.7                   | 100%                  | 474.5            | 100%             |
| 2.00 | 6.7                   | 100%                  | 225.1            | 100%             |
| 5.00 | 6.7                   | 100%                  | 29.6             | 100%             |

### Analysis

1. **Perceptron:** The perceptron is relatively insensitive to the learning rate. All values of eta achieved 100% convergence. Very small eta (0.01) requires slightly more epochs because the weight updates are smaller, but the perceptron always converges quickly for linearly separable problems.

2. **Delta Learning:** The delta rule is highly sensitive to the learning rate:
   - **eta = 0.01:** Failed to converge within 10,000 epochs. The updates are too small to push the sigmoid outputs close enough to +/-1.
   - **eta = 0.1:** Converged but required ~5,000 epochs on average.
   - **eta = 5.0:** Converged in only ~30 epochs, a 170x improvement over eta = 0.1.
   - The relationship between eta and convergence speed is roughly inverse for the delta rule.

3. **Key Insight:** For the delta rule, larger learning rates dramatically accelerate convergence because the sigmoid derivative `(1-o^2)/2` is small near saturation, creating a "vanishing gradient" effect. A larger eta compensates for this small gradient. However, excessively large eta can cause divergence in more complex problems.

![Learning Rate Comparison](Practice/Part1/eta_comparison.png)

*Figure 2: Learning rate comparison for AND gate. Top: error curves. Bottom-left: average convergence epochs (log scale). Bottom-right: convergence rate.*

---

## f) Discussion on Extended Experiments

### Summary of Findings

The extended experiments in d) and e) reveal two important aspects of single-neuron learning:

1. **Fundamental Limitation (XOR):** The XOR experiment conclusively demonstrates that a single neuron, regardless of the learning rule or activation function, cannot solve non-linearly separable problems. The perceptron's error oscillates without decreasing, while the delta rule's error stagnates at a local minimum. This limitation motivated the development of multi-layer neural networks and the backpropagation algorithm.

2. **Hyperparameter Sensitivity:** The learning rate comparison shows that while the perceptron is robust to eta choices (due to its discrete output), the delta rule requires careful tuning. Too small an eta leads to extremely slow convergence or failure, while larger values dramatically speed up learning. This highlights the importance of hyperparameter tuning in gradient-based methods and foreshadows challenges in training deep neural networks.

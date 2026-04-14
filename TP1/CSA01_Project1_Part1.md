# CSA01 - Team Project 1, Part 1: Single Neuron Learning

**Course:** Neural Networks (CSA01)  
**Team Members:**
- SATO Sho (m5301059)
- USAMI Yuki (m5301073)
- SEKINE Kento (m5301060)
- AIZAWA Yuma (m5301001)
- WATABE Chitose (m5301074)

---

## a) Problem Description

In Part 1, we implemented two learning rules for a single neuron — the perceptron learning rule and the delta learning rule — and compared how they behave on a simple classification task.

We used the **AND gate** as our target problem. It takes two binary inputs and outputs +1 only when both are 1. Since the AND function is linearly separable, a single neuron should be able to learn it. The training data is as follows (the third input is a fixed bias of -1):

| Input x1 | Input x2 | Bias | Desired Output d |
|:---------:|:---------:|:----:|:----------------:|
| 0         | 0         | -1   | -1               |
| 0         | 1         | -1   | -1               |
| 1         | 0         | -1   | -1               |
| 1         | 1         | -1   | 1                |

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

The perceptron uses a step function as its activation:

```
f(x) = +1  if x >= 0
f(x) = -1  if x < 0
```

Weights are updated each time a pattern is presented:

```
w_i(t+1) = w_i(t) + eta * (d - o) * x_i
```

Here, `eta = 0.5`, `d` is the desired output, `o` is the actual output, and `x_i` is the input. The error is `E = 0.5 * (d - o)^2`, and training stops when the total error over all 4 samples drops below 0.01.

### Delta Learning Rule

The delta rule uses a sigmoid activation scaled to [-1, 1]:

```
f(x) = 2 / (1 + exp(-lambda * x)) - 1,  lambda = 1.0
```

The weight update uses the derivative of the sigmoid:

```
delta = (d - o) * (1 - o^2) / 2
w_i(t+1) = w_i(t) + eta * delta * x_i
```

The convergence condition is the same (total error < 0.01).

### Common Setup

- 3 inputs (2 data + 1 bias)
- Learning rate: 0.5
- Weights initialized randomly in [0, 1)

## c) Discussion of Simulation Results

### Perceptron Learning Results

The perceptron converged in **5 epochs**. Final weights:

```
w = [1.528, 1.227, 2.427]
```

| Input       | Output | Expected |
|:-----------:|:------:|:--------:|
| (0, 0, -1)  | -1     | -1       |
| (0, 1, -1)  | -1     | -1       |
| (1, 0, -1)  | -1     | -1       |
| (1, 1, -1)  | +1     | +1       |

All outputs matched perfectly.

### Delta Learning Results

The delta rule took **969 epochs** to converge. Final weights:

```
w = [6.281, 6.278, 9.503]
```

| Input       | Output    | Expected |
|:-----------:|:---------:|:--------:|
| (0, 0, -1)  | -0.9999   | -1       |
| (0, 1, -1)  | -0.9235   | -1       |
| (1, 0, -1)  | -0.9233   | -1       |
| (1, 1, -1)  | +0.9101   | +1       |

The outputs are close to the targets, but not exact. The sigmoid never actually reaches +/-1, so the outputs are always approximate.

### Comparison

The most obvious difference is speed: the perceptron finished in 5 epochs while the delta rule needed 969. This makes sense — the perceptron's step function gives exact +1/-1 outputs, so once the decision boundary is in the right place, the error immediately hits zero. The delta rule's sigmoid outputs only approach +/-1 asymptotically, so the weights have to keep growing to push the outputs closer to the targets. That is also why the delta rule's final weights (6~9) are much larger than the perceptron's (1~2).

Another thing we noticed is how the error decreases differently. The perceptron error jumps in steps of 2 (because `(d-o)^2 = 4` when the output is wrong), while the delta rule's error decreases smoothly. This reflects the continuous vs. discrete nature of the two activation functions.

On the other hand, the delta rule has the advantage of using a differentiable activation function. For a simple problem like AND, this does not matter, but it becomes necessary for training multi-layer networks with backpropagation.

---

## d) New Problem: XOR Gate

We tried applying both learning rules to the **XOR problem** to see what happens when the problem is not linearly separable.

| Input x1 | Input x2 | Desired Output d |
|:---------:|:---------:|:----------------:|
| 0         | 0         | -1               |
| 0         | 1         | +1               |
| 1         | 0         | +1               |
| 1         | 1         | -1               |

We ran both rules for up to 1,000 epochs. Neither converged:

- **Perceptron:** error stayed at 8.0 — two patterns are always wrong, and each wrong pattern gives an error of 4.0.
- **Delta rule:** error settled around 2.59 — the outputs drifted toward 0, not being able to commit to either +1 or -1.

This result is expected. A single neuron draws one straight line as its decision boundary (`w1*x1 + w2*x2 = threshold`), but XOR cannot be separated by a single line. The +1 class (inputs (0,1) and (1,0)) and the -1 class (inputs (0,0) and (1,1)) are diagonally opposite, so no matter how you draw one line, it will always misclassify at least one pattern. This is a well-known limitation of single-layer networks.

![XOR Error Curves](Practice/Part1/xor_error.png)

*Figure 1: Error curves for XOR. The perceptron error stays flat at 8.0, and the delta rule error plateaus around 2.6.*

---

## e) Modification: Learning Rate Comparison

We tested how the learning rate `eta` affects convergence on the AND gate. We tried 6 values (0.01, 0.1, 0.5, 1.0, 2.0, 5.0) and ran 10 trials with different random seeds for each.

| eta  | Perceptron Avg Epochs | Perceptron Conv. Rate | Delta Avg Epochs | Delta Conv. Rate |
|:----:|:---------------------:|:---------------------:|:----------------:|:----------------:|
| 0.01 | 9.5                   | 100%                  | N/A              | 0%               |
| 0.10 | 3.5                   | 100%                  | 4947.5           | 100%             |
| 0.50 | 6.1                   | 100%                  | 971.6            | 100%             |
| 1.00 | 6.7                   | 100%                  | 474.5            | 100%             |
| 2.00 | 6.7                   | 100%                  | 225.1            | 100%             |
| 5.00 | 6.7                   | 100%                  | 29.6             | 100%             |

The perceptron converged at all learning rates and was not very sensitive to eta. The delta rule, on the other hand, varied a lot. At eta = 0.01, it could not converge within 10,000 epochs at all. At eta = 5.0, it finished in about 30 epochs — roughly 170 times faster than eta = 0.1.

We think this is because the sigmoid derivative `(1-o^2)/2` gets very small when the output is near +/-1. With a tiny eta on top of that, the weight updates become too small to make progress. A larger eta compensates for this effect. Of course, in more complex problems, too large an eta could cause the training to diverge, but for this simple AND problem, bigger was better.

![Learning Rate Comparison](Practice/Part1/eta_comparison.png)

*Figure 2: Learning rate comparison. Top: error curves. Bottom-left: average epochs to converge (log scale). Bottom-right: convergence rate.*

---

## f) Discussion on Extended Experiments

From the XOR experiment, we confirmed that a single neuron simply cannot solve a non-linearly separable problem, no matter how long we train it. This is the fundamental reason why multi-layer networks were developed.

From the learning rate experiment, we learned that the delta rule is much more sensitive to eta than the perceptron. The perceptron always converges quickly regardless of eta (for linearly separable problems), but the delta rule can be extremely slow or even fail to converge if eta is too small. Choosing the right learning rate matters a lot for gradient-based methods.

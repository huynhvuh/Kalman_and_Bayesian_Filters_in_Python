# Kalman and Bayesian Filters in Python

- Link: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

- Setup Conda environment:
  - conda create -n KBF
  - conda install -c miniconda jupyter numpy matplotlib scipy sympy
  - pip install filterpy


## 01. g-h filter
- Algorithm
    - Initialization
        1. Initialize the state of the filter
        2. Initialize our belief in the state
    - Predict
        1. Use system behavior to predict state at the next time step
        2. Adjust belief to account for the uncertainty in prediction
    - Update
        1. Get a measurement and associated belief about its accuracy
        2. Compute residual between estimated state and measurement
        3. New estimate is somewhere on the residual line

- Benedict-Bordner filter was invented to minimize the transient error in this example, where the derivative of x makes a step jump


## 02. Discrete Bayes Filter
- Key idea: `posterior = (likelihood * prior) / normalization factor`

- Algorithm
    - Initialization
        1. Initialize our belief in the state
    - Predict
        1. Based on the system behavior, predict state for the next time step
        2. Adjust belief to account for the uncertainty in prediction
    - Update
        1. Get a measurement and associated belief about its accuracy
        2. Compute how likely it is the measurement matches each state
        3. Update state belief with this likelihood 

- The filter equations:
    - Predict step:   $x_{prior} = x_{posterior} f_x(.)$
    - Update step:    $x_{posterior} = || L x_{prior} ||$


## 04. One Dimensional Kalman Filters
- The Kalman filter is a Bayesian filter that uses Gaussians, with the objective to replace the histogram with a Gaussian to reuduce computations. (Instead of computating millions of values and probability of histogram, use only 2 values of mean and variance )

- Modeling the performance of your sensors is one of the harder parts of creating a Kalman filter that performs well.

- Algorithm
    - Initialization
        1. Initialize the state of the filter
        2. Initialize our belief in the state
    - Predict
        1. Use system behavior to predict state at the next time step
        2. Adjust belief to account for the uncertainty in prediction
    - Update
        1. Get a measurement and associated belief about its accuracy
        2. Compute residual between estimated state and measurement
        3. Compute scaling factor based on whether the measurement or prediction is more accurate
        4. set state between the prediction and measurement based on scaling factor
        5. update belief in the state based on how certain we are in the measurement


## 06. Multivariate Kalman Filters
- Equations:
    - Predict step:
        $\bar{𝐱} = 𝐅 𝐱 + 𝐁 𝐮$
        $\bar{𝐏} = F 𝐏 𝐅^T + 𝐐$
        Where,
            - $𝐱$, $𝐏$ are the state mean and covariance. They correspond to $𝑥$ and $\sigma^2$.
            - $𝐅$ is the state transition function. When multiplied by $𝐱$ it computes the prior.
            - $𝐐$ is the process covariance. It corresponds to $\sigma_{𝑓𝑥}$.
            - $𝐁$ and $𝐮$ are new to us. They let us model control inputs to the system.
    - Update step:
        $𝐲 = 𝐳 - 𝐇 \bar{𝐱}$
        $𝐊 = \bar{𝐏} 𝐇^T (𝐇 \bar{𝐏} 𝐇^T + 𝐑)^{-1}$
        $𝐱 = \bar{𝐱} + 𝐊 𝐲$
        $𝐏 = (𝐈 - 𝐊 𝐇) \bar{𝐏}$
        - Where:
            - $𝐇$ is a measurement function that converts a state into a measurement 
              (In case that $𝐳$ and $\bar{𝐱}$ are not in a same unit). Why are we working in measurement space and not in state space? $\rightarrow$ We cannot do that because most measurements are not invertible
            - $𝐳, 𝐑$ are the measurement mean and noise covariance. They correspond to $𝑧$ and $\sigma^2_𝑧$ in the univariate filter. (I've substituted $\mu$ with $𝑥$ for the univariate equations to make the notation as similar as possible).
            - $𝐲$ and $𝐊$ are the residual and Kalman gain.


## 07. Kalman Filter Math
- 3 common ways to find $𝐅$ from $𝐀$:
    - Matrix exponential technique
    - Linear Time Invariant Theory (LTI Theory)
    - Numerical techniques


## 08. Designing Kalman Filters
- Steps:
    1) Choose the State Variables: $𝐱 = [ 𝑥 \; \dot{x} \; 𝑦 \; \dot{y} ]^T$ is better than $𝐱 = [ 𝑥 \; 𝑦 \; \dot{x} \; \dot{y} ]^𝖳$
    2) Design State Transition Function: $𝐅$ where $\bar{𝐱} = 𝐅 𝐱$
    3) Design the Process Noise Matrix: $𝐐$
    4) Design the Control Function: $𝐁$
    5) Design the Measurement Function: $𝐇$ where $𝐳 = 𝐇 𝐱$
    6) Design the Measurement Noise Matrix: $𝐑$
    7) Initial Conditions: $𝐱$ and $𝐏$
    8) Implement the Filter

- Important Topics:
    - Filter Order
    - Evaluating Filter Order: 
        - Look at the residuals between the estimated state and actual state and compare them to the standard deviations which we derive from $𝐏$
          For simulated systems: the filter is performing correctly if 99% of the residuals will fall within $3\sigma$
          For real sensors: the filter is performing correctly if 99% of the residuals will fall within $5\sigma$ (for example)
        - Important note: 
            - The covariance matrix $𝐏$ is only reporting the theoretical performance of the filter assuming all of the inputs are correct
              Possible case: Smaller $𝐏$ implies that the Kalman filter's estimates are getting better and better with time while the Kalman filter is diverging
              $\rightarrow$ `Smug filter` means that the filter is overconfident in its performance
              *** Do not trust the filter's covariance matrix to tell you if the filter is performing well!
    - Detecting and Rejecting Bad Measurement $\rightarrow$ Gating
        - A gate is a formula or algorithm that determines if a measurement is good or bad. Only good measurements get through the gate
            - Rectangular gate: Gate limits (number of standard deviations)
                - Ex: residual_x > Gate_limit * std_x
                - Cheap computation but expensive computation in higher dimensions
            - Ellipsoidal gate: Mahalanobis distance
                - The mahalanobis distance 𝐷𝑚 is a statistical measure of the distance of a point from a distribution
                - 𝐷𝑚 = standard deviation distance point to a distribution
                - Expensive computation in high dimensions
            - Maneuver gate: 
                - Defines a region where it is possible for the object to maneuver, taking the object's current velocity and ability to maneuver into account
                - Quite cheap computation
            - 2-gate approach:
                - Gate 1 = large and rectangular gate $\rightarrow$ throw away obviously bad measurements
                - Gate 2 = ellipsoidal gate
    - Gating and Data Association Strategies
        - Data Association: 
            - Ex: The radar detects multiple signals on each sweep. 
                  Suppose the first sweep gets 5 measurements. We would create 5 potential tracks.
                  In the next sweep we get 6 measurements.
                  We can combine any of the first measurements with any of the second measurements, giving us 30 potential tracks.
                  After just a few epochs we reach millions, then billions of potential tracks.
        - Tracking is a matter of gating, data association, and pruning
            - Solution: Particle Filter which solves the combinatorial explosion with statistical sampling
        - Books: 
            - Multiple-target Tracking with Radar Applications
            - Fundamentals of Object Tracking
            - Bayesian Multiple Target Tracking
    - Evaluating Filter Performance:
        - Normalized Estimated Error Squared (NEES): Chi-squared distributed with n degrees of freedom
            - Take the average of all the NEES values, and they should be less than the dimension of x
        - Likelihood Function or Log-likelihood:
            - Distinguish:
                - A probability is the chance of something happening - as in what is the probability that a fair die rolls 6 three times in five rolls?
                - The likelihood asks the reverse question - given that a die rolled 6 three times in five rolls, what is the likelihood that the die is fair?
            - The likelihood is a measure of how likely the measurement are given the current state.
              If the likelihood is low we know one of our assumptions is wrong
    - Control Inputs
    - Sensor Fusion:
        - Redesign the matrix $𝐇$ to convert $𝐱$ to $𝐳$
        - Redesign the measurement noise matrix $𝐑$
        - Important topic: 
            - [`Iterative Least Squares for Sensor Fusion`](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Supporting_Notebooks/Iterative-Least-Squares-for-Sensor-Fusion.ipynb)
            - GPS sensors cannot be filtered by Kalman filters, because:
                - Kalman filter must be Gaussian and time independent while the output of the GPS is time dependent 
                  because the filter bases its current estimate on the recursive estimates of all previous measurements
                - GPS signals are already optimal
            - Sensor fusion of position sensor and wheel measurements yields a better result than using the wheel alone.
        - Different Data Rates:
            - Solution: 
                1) Wait for a data packet from either sensor
                2) Determine the amount of time that has passed since the last update
                3) Modify the affected matrices $𝐅$, $𝐐$, $𝐇$ and $𝐑$ on every innovation
    - Nonstationary Processes: 
        - Matrices changes over time
        - What if our data rate changes in some unpredictable manner? Or what if we have two sensors, each running at a different rate?
          $\rightarrow$ Alter the Kalman filter matrices to reflect the current situation (Change dt for each epoch)
        - How would we handle Δ𝑡 changing for each measurement?
          $\rightarrow$ Modify the relevant matrices


## 09. Nonlinear Filtering
- `You can only design a filter after understanding the particular problems the nonlinearity in your problem causes`

- Book:
    - Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches

- The Algorithms:
    - The linearized Kalman filter and extended Kalman filter (EKF):
        - Extreme demanding due to finding a solution to a matrix of partial derivatives (Jacobian matrix)
        - Analytical techniques of solving are usually difficult or impossible
        - Innovation: Numerical techniques using Monte Carlo techniques
            - Robust for extremely nonlinear problems
            - Able to tracj many objects at once
    - The UKF:
        - It is superior in almost every way to the EKF
        - It can be a few times slower than the EKF, this really depends on:
            - Whether the EKF solves the Jacobian analytically or numerically
            - The UKF dispenses with the need to find solutions to partial differential equations
        - Numerically the UKF is almost certainly faster than the EKF
    - Particle filter:
        - It dispenses with mathematical modeling completely in favor of a Monte Carlo technique of generating a random cloud of thousands of points
        - It runs slowly
        - It can solve intractable problems with relative ease
    - Balancing issues:
        - Accuracy
        - Round off errors
        - Divergence
        - Mathematical proof of correctness
        - Computational effort required

# JPM MLCOE Time Series & Reinforcement Learning Internship

This repository contains the implementation of Questions 2 and 3.

- Q2: Particle Flow Filter and Differentiable Particle Filter
- Q3: Discrete Choice Model and Credit Card Offers

~~The Q3 implementation is designed to be plug-and-play with the `choice-learn` library.~~

Q3 implementation now supports poetry!

## Project Structure

```
Q2/
├── Q2/                     # TensorFlow/TFP-based implementation  
│   ├── experiments/        # Experiment scripts
│   ├── src/                # Core modules (filters, flows, ssm, resampling, utils)
│   └── tests/              # Unit / integration tests
└── Q2_np/                  # NumPy-based implementation (Part 1 submission)
    ├── experiments/
    ├── src/
    └── tests/

Q3/
├── datasets/               # Data Generating Process (DGP)
├── experiments/            # Experiment scripts
│   └── utils/
├── models/                 # Choice models
└── tests/                  # Unit / integration tests
```

If you need any other information or clarification, please contact: haeun39@kaist.ac.kr

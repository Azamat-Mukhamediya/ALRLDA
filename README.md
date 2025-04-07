# Efficient Active Learning using Recursive Estimation of Error Reduction

This repository is the official implementation of [Efficient Active Learning using Recursive Estimation of Error Reduction](https://www.sciencedirect.com/science/article/pii/S0925231225008033).

## Requirements

This code has been developed under `Python 3.9.16` and `scikit-learn 1.3.0` on `MacOS 13.1 Ventura`.

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

To reproduce the results presented in the paper, run this command:

```
python main.py --data hlc --label_size 100 --test_size 0.2 --budget_size 10 --cycles 20 --model RLDA --sampling ALRLDA
```

## Data

Datasets are colleclted from OpenML(https://openml.org/) dataset repository.

## Acknowledgments

We would like to acknowledge the following GitHub repositories for providing their codebase:

- https://github.com/clabrugere/pytorch-scarf
- https://github.com/avihu111/TypiClust

## Citation

```

@article{MukhamediyaALRLDA,
	author = {Azamat Mukhamediya and Reza Sameni and Amin Zollanvari},
	doi = {https://doi.org/10.1016/j.neucom.2025.130131},
	issn = {0925-2312},
	journal = {Neurocomputing},
	keywords = {Active learning, Estimated error reduction, Regularized Linear Discriminant Analysis},
	pages = {130131},
	title = {Efficient active learning using recursive estimation of error reduction},
	url = {https://www.sciencedirect.com/science/article/pii/S0925231225008033},
	volume = {638},
	year = {2025},
	bdsk-url-1 = {https://www.sciencedirect.com/science/article/pii/S0925231225008033},
	bdsk-url-2 = {https://doi.org/10.1016/j.neucom.2025.130131}
}

```

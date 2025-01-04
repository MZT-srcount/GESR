---
pretty_name: SRSD-Feynman (Medium)
annotations_creators:
- expert
language_creators:
- expert-generated
language:
- en
license:
- cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
source_datasets:
- extended
task_categories:
- tabular-regression
task_ids: []
---

# Dataset Card for SRSD-Feynman (Medium set)

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:**
- **Repository:** https://github.com/omron-sinicx/srsd-benchmark
- **Paper:** [Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery](https://arxiv.org/abs/2206.10540)
- **Point of Contact:** [Yoshitaka Ushiku](mailto:yoshitaka.ushiku@sinicx.com)

### Dataset Summary

Our SRSD (Feynman) datasets are designed to discuss the performance of Symbolic Regression for Scientific Discovery.
We carefully reviewed the properties of each formula and its variables in [the Feynman Symbolic Regression Database](https://space.mit.edu/home/tegmark/aifeynman.html) to design reasonably realistic sampling range of values so that our SRSD datasets can be used for evaluating the potential of SRSD such as whether or not an SR method con (re)discover physical laws from such datasets.

This is the ***Medium set*** of our SRSD-Feynman datasets, which consists of the following 40 different physics formulas:

[![Click here to open a PDF file](problem_table.png)](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium/resolve/main/problem_table.pdf)


More details of these datasets are provided in [the paper and its supplementary material](https://openreview.net/forum?id=qrUdrXsiXX).  

### Supported Tasks and Leaderboards

Symbolic Regression

## Dataset Structure

### Data Instances

Tabular data + Ground-truth equation per equation

Tabular data: (num_samples, num_variables+1), where the last (rightmost) column indicate output of the target function for given variables.
Note that the number of variables (`num_variables`) varies from equation to equation.
  
Ground-truth equation: *pickled* symbolic representation (equation with symbols in sympy) of the target function.


### Data Fields

For each dataset, we have 
1. train split (txt file, whitespace as a delimiter)
2. val split (txt file, whitespace as a delimiter)
3. test split (txt file, whitespace as a delimiter)
4. true equation (pickle file for sympy object)

### Data Splits

- train: 8,000 samples per equation
- val: 1,000 samples per equation
- test: 1,000 samples per equation

## Dataset Creation

### Curation Rationale

We chose target equations based on [the Feynman Symbolic Regression Database](https://space.mit.edu/home/tegmark/aifeynman.html).

### Annotations

#### Annotation process

We significantly revised the sampling range for each variable from the annotations in the Feynman Symbolic Regression Database.
First, we checked the properties of each variable and treat physical constants (e.g., light speed, gravitational constant) as constants.
Next, variable ranges were defined to correspond to each typical physics experiment to confirm the physical phenomenon for each equation.
In cases where a specific experiment is difficult to be assumed, ranges were set within which the corresponding physical phenomenon can be seen.
Generally, the ranges are set to be sampled on log scales within their orders as 10^2 in order to take both large and small changes in value as the order changes.
Variables such as angles, for which a linear distribution is expected are set to be sampled uniformly.
In addition, variables that take a specific sign were set to be sampled within that range.

#### Who are the annotators?

The main annotators are
- Naoya Chiba (@nchiba)
- Ryo Igarashi (@rigarash)



### Personal and Sensitive Information

N/A

## Considerations for Using the Data

### Social Impact of Dataset

We annotated this dataset, assuming typical physical experiments. The dataset will engage research on symbolic regression for scientific discovery (SRSD) and help researchers discuss the potential of symbolic regression methods towards data-driven scientific discovery.

### Discussion of Biases

Our choices of target equations are based on [the Feynman Symbolic Regression Database](https://space.mit.edu/home/tegmark/aifeynman.html), which are focused on a field of Physics.

### Other Known Limitations

Some variables used in our datasets indicate some numbers (counts), which should be treated as integer.
Due to the capacity of 32-bit integer, however, we treated some of such variables as float e.g., number of molecules (10^{23} - 10^{25})

## Additional Information

### Dataset Curators

The main curators are
- Naoya Chiba (@nchiba)
- Ryo Igarashi (@rigarash)

### Licensing Information

Creative Commons Attribution 4.0

### Citation Information

[[OpenReview](https://openreview.net/forum?id=qrUdrXsiXX)] [[Video](https://www.youtube.com/watch?v=MmeOXuUUAW0)] [[Preprint](https://arxiv.org/abs/2206.10540)]  
```bibtex
@article{matsubara2024rethinking,
  title={Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery},
  author={Matsubara, Yoshitomo and Chiba, Naoya and Igarashi, Ryo and Ushiku, Yoshitaka},
  journal={Journal of Data-centric Machine Learning Research},
  year={2024},
  url={https://openreview.net/forum?id=qrUdrXsiXX}
}
```

### Contributions

Authors:
- Yoshitomo Matsubara (@yoshitomo-matsubara)
- Naoya Chiba (@nchiba)
- Ryo Igarashi (@rigarash)
- Yoshitaka Ushiku (@yushiku)



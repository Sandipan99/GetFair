# GetFair: Generalized Fairness Tuning of Classification Models

> GetFair: Generalized Fairness Tuning of Classification Models. Sandipan Sikdar, Florian Lemmerich and Markus Strohmaier accepted at [ACM FAccT 2022](https://facctconference.org/2022/index.html)

***Please cite our paper in any published work that uses any of these resources.***

~~~
Coming Soon
~~~

## Abstract
We present GetFair, a novel framework for tuning fairness of classification models. The fair classification problem deals with training
models for a given classification task where data points have sensitive attributes. The goal of fair classification models is to not only
generate accurate classification results but also to prevent discrimination against sub populations (i.e., individuals with a specific value
for the sensitive attribute). Existing methods for enhancing fairness of classification models however are often specifically designed
for a particular fairness metric or a classifier model. They may also not be suitable for scenarios with incomplete training data or
where optimizing for multiple fairness metrics is important. GetFair represents a general solution to this problem.
The GetFair approach works in the following way: First, a given classifier is trained on training data without any fairness objective.
This is followed by a reinforcement learning inspired tuning procedure which updates the parameters of the learnt model on a given
fairness objective. This disentangles classifier training from fairness tuning which makes our framework more general and allows
for adopting any parameterized classifier model. Because fairness metrics are designed as reward functions during tuning, GetFair
generalizes across any fairness metric.
We demonstrate the generalizability of GetFair via evaluation over a benchmark suite of datasets, classification models and fairness
metrics. In addition, GetFair can also be deployed in settings where the training data is incomplete or the classifier needs to be tuned
on multiple fairness metrics. GetFair not only contributes a flexible method to the repertoire of tools available for enhancing fairness of
classification models, it also seamlessly adapts to settings where existing fair classification methods may not be suitable or applicable

## Requirements
1. sklearn 0.23.1
2. numpy 1.18.5
3. Pytorch 1.6.0
4. scipy 1.6.0
5. click 7.1.2
6. tqdm 4.47.0

## Running the code

To tune for single metric, execute -
```
python main.py -d <dataset> -f <metric> -t <tuning rate> -c <classifier>
```
```
<dataset>
Synthetic Statistical Parity (1) (default) 
Sythetic Equal Opportunity (2)
Synthetic Equalized odds (3)
Adult (5)
Compas (6)
Bank (7)
```
```
<metric>  
"stpr" - statistical parity (default)
"eoppr" - equal opportunity
"eodd" - equalized odds
```
An execution instance -
```
python main.py -d 1 -f stpr -t 1 -c log-reg
```
```
tuning rate - any floating point value between (0 and 1] (default-1)
```
```
<classifier>
"log-reg" - logistic regression (default)
"lin-svm" - linear SVM
"mlp" - neural network
```
To tune for both statistical parity and equal opportunity execute - 
```
python multiple_metric.py -d <dataset> -c <classifier>
```

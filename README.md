# Machine-learning modeling of lung mechanics: assessing the variability and propagation of uncertainty in pulmonary compliance and resistance

This is the front page README.md

## Abstract
In mechanical ventilation therapy, respiratory mechanics is traditionally evaluated in terms of the respiratory-system
compliance and airways resistance. Clinical evidence has shown that these physiological parameters present a large
variability and heterogeneity among patients, which motivates the creation of models that can efficiently predict the
lung function to personalize ventilator settings during treatment. In this work, we leverage machine learning tech-
niques to accelerate the predictions of lung mechanics, to further study the lung response via uncertainty analysis. To
this end, we implement a high-fidelity anatomical lung model based on computed-tomography images, derived from
a continuum poromechanical framework. Our formulation considers parameters related to the constitutive lung tissue
model, the parenchyma permeability, and the effect of chest wall stiffness in the estimation of respiratory-system com-
pliance and resistance. Based on this, we develop a low-fidelity model and integrate both levels of information to train
a surrogate machine learning lung model based on multi-fidelity Gaussian process regression, comparing the performance
with single-fidelity Gaussian process and artificial neural networks. Once trained and validated, we perform
a parameter sensitivity analysis and uncertainty quantification tasks on the lung response. In this sense, our results
suggest that lung tissue elasticity parameters and the chest wall stiffness have a comparable level of influence on the
respiratory-system compliance, while the airways resistance is mainly modulated by the parenchyma permeability,
with both physiological variables showing non-linear relations. Regarding the machine learning methodology, the
constructed multi-fidelity GP surrogate model outperformed the single-fidelity model and the neural network, high-
lighting the desirability of adopting approaches that combine different levels of fidelity to obtain accurate and efficient
estimates in biomechanical simulations. We expect that the machine learning methods and results shown here will be
valuable for the generation of lung models, and also, in the application of similar biomechanical models to support
relevant clinical situations

## Directories
- `results-data`: Figures and related results from processed from raw data and tests.
- `raw-data`: Data generated from code files.
- `tests`:  Code files that implement the described work.

## Results
Fig 1
Fig 2
...

## Dependencies
- `FeniCS` 2019.1.0
- `numpy`
- `scipy`
- `matplotlib`
- `GPy`
- `Emukit`
- `os`
- `sys`

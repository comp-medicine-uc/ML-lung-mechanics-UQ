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
- `results-data`: Figures and related results from processed from `raw-data` and `tests`.
- `raw-data`: Input data needed for simulations + data generated from `tests`.
- `tests`:  Code files that implement the described work.

## Results
The main results of this research are shown below. Figure 3 shows the airway pressure, flow and volume signals predicted by high and low fidelity models of the lung during PCV simulations.
![Fig3_HFvsLF](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/f1ecc5a9-ad89-4a33-9df5-299a5821bd7c)


Figure 3. Simulation of lungs under PCV ventilation model. Physiological signals that describe the time evolution of (a) airways pressure, (b) flow, and (c) volume (c) are shown for the high-fidelity (solid lines) and low-fidelity model (dashed lines).

![Fig4_blandaltman_c](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/f8e77dc1-d411-42f2-a1ec-3bbac93653cb)

Figure 4. Simulation of lungs under PCV ventilation model. Physiological signals that describe the time evolution of (a) airways pressure, (b) flow, and (c) volume (c) are shown for the high-fidelity (solid lines) and low-fidelity model (dashed lines).

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

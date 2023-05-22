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


![Fig6_rmse](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/f3277706-8cf7-43bf-becd-5864ec6d6d77)

Figure 4. Effect of the training dataset size on the prediction performance of the multi-fidelity GP, single-fidelity GP, and neural network models. (a) Respiratory-system compliance, and (b) Resistance. Results shown are for the testing set predictions. Dashed lines and error bars denote the average and standard deviation of the RMSE, respectively.


![Fig4_blandaltman_c](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/f8e77dc1-d411-42f2-a1ec-3bbac93653cb)

Figure 5. Performance comparison of multi-fidelity GP, single-fidelity GP, and neural network models on the respiratory-system compliance. The predictions on the testing set are analyzed using (a) correlation plots, and (b) Bland-Altman plots.


![Fig7_sobol_indices](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/ba12b571-e853-4e24-932b-3f2f8381d7ce)

Figure 6. Sobol total sensitivity indices with respect to lung model parameters [$c$, $\beta$, $c_1$, $c_3$, $k$, $K_s$], for (a) respiratory-system compliance, and (b) airways resistance. Each bar indicates the total-order index, while error bars indicate the 95\% confidence intervals.


![Fig8_maineffects_c](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/6ce37804-a56e-465c-938f-efd4dc67bbb0)

Figure 7. Main effects analysis of the respiratory-system compliance. Parameters $c$, $\beta$ y $K_s$ show an inverse relationship with compliance. Furthermore, $c$ and $K_s$ have the greatest influence on the response. On the other hand, $k$ shows a slightly direct relation with compliance. Parameters $c_1$ and $c_3$ have no influence on the response. Gray lines represent the 100 trajectories used for the simulation, while black lines represent the average main effect.


![Fig10_uncertainty_example](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/ad03baa5-c0ad-43fe-96ec-d17148f1448a)

Figure 8. Uncertainty propagation of constitutive model parameter $c$ and its effect on the response variability. This example case considers a $\pm$25\% uncertainty respect to the baseline value, where (a) corresponds to the uniform probability distribution of parameter $c$, which is applied to our surrogate model, obtaining in (b) empirical distributions for respiratory system-compliance and resistance response. The dashed vertical line represents the baseline value for $c$ and the, while solid vertical lines represent the reference response of the model for the parameters baseline values. Solid curves represent the corresponding probability density functions.


![Fig11_parameter_uncertainty](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/d2d9d94c-a2b9-4e82-b779-8b9b1782f8c5)

Figure 9. Response variability in respiratory-system compliance, and airways resistance with respect to three levels of parameter uncertainty: $\Delta = \pm10\%, \pm25\%, \pm50\%$. Each row corresponds to one of the six parameters [$c$, $\beta$, $c_1$, $c_3$, $k$, $K_s$]. An additional row shows the variability when uncertainty is present in all parameters simultaneously (All). Vertical lines are the reference response of the model for the baseline values of the parameters.

## Dependencies
- `FeniCS` 2019.1.0
- `numpy`
- `scipy`
- `matplotlib`
- `GPy`
- `Emukit`
- `os`
- `sys`

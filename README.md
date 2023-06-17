# Machine learning modeling of lung mechanics: assessing the variability and propagation of uncertainty in respiratory-system compliance and airway resistance

Leverage of machine learning techniques to construct predictive lung function models informed by finite element simulations.

## Abstract
The response of patients to mechanical ventilation is traditionally evaluated in terms of respiratory-system compliance and airway resistance. Clinical evidence has shown high variability in these parameters, highlighting the difficulty to predict them before the start of ventilation therapy. This motivates the creation of computational models that can connect structural and tissue features with lung mechanics. Our objective is to leverage machine learning (ML) techniques to construct predictive lung function models that are informed by non-linear finite element simulations. To this end, we revisit a continuum poromechanical formulation of the lungs suitable for determining respiratory-system compliance and airway resistance. Based on this framework, we create high-fidelity non-linear finite element models of human lungs from medical images. We also develop a low-fidelity model based on an idealized sphere geometry. We then use these models to train and validate three ML architectures: single-fidelity and multi-fidelity Gaussian process regression, and artificial neural networks. We use the best ML model in the prediction of lung function and to further study the sensitivity of lung mechanics to variations in tissue structural parameters and boundary conditions via sensitivity analysis and forward uncertainty quantification. The low-fidelity model delivers a lung response very close to that predicted by high-fidelity simulations and at a fraction of the computational time. Regarding the trained ML models, the multi-fidelity GP model consistently delivers better accuracy than the single-fidelity GP and neural network models in estimating respiratory-system compliance and resistance ($\sim R^2$). In terms of computational efficiency, our ML model delivers a massive speed-up of $\sim970,000\times$ with respect to the high-fidelity simulations. Regarding lung function, we observed an almost matched and non-linear behavior between some of the structural parameters and chest wall stiffness with compliance. Also, we observed a strong modulation of airways resistance with tissue permeability. Our findings suggest the relevance of certain constitutive lung tissue model parameters and boundary conditions in the respiratory-system response and airway resistance. Furthermore, we highlight the advantages of adopting a multi-fidelity ML approach that combines data from different levels to obtain accurate and efficient estimates in biomechanical simulations. We expect that the methods and results shown here will be valuable for generating lung models and similar organs to support relevant clinical situations.

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

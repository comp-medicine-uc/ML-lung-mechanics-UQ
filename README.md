# Machine learning modeling of lung mechanics: assessing the variability and propagation of uncertainty in respiratory-system compliance and airway resistance

Leverage of machine learning techniques to construct predictive lung function models informed by finite element simulations.

## Abstract
The response of patients to mechanical ventilation is traditionally evaluated in terms of respiratory-system compliance and airway resistance. Clinical evidence has shown high variability in these parameters, highlighting the difficulty to predict them before the start of ventilation therapy. This motivates the creation of computational models that can connect structural and tissue features with lung mechanics. Our objective is to leverage machine learning (ML) techniques to construct predictive lung function models that are informed by non-linear finite element simulations. To this end, we revisit a continuum poromechanical formulation of the lungs suitable for determining respiratory-system compliance and airway resistance. Based on this framework, we create high-fidelity non-linear finite element models of human lungs from medical images. We also develop a low-fidelity model based on an idealized sphere geometry. We then use these models to train and validate three ML architectures: single-fidelity and multi-fidelity Gaussian process regression, and artificial neural networks. We use the best ML model in the prediction of lung function and to further study the sensitivity of lung mechanics to variations in tissue structural parameters and boundary conditions via sensitivity analysis and forward uncertainty quantification. The low-fidelity model delivers a lung response very close to that predicted by high-fidelity simulations and at a fraction of the computational time. Regarding the trained ML models, the multi-fidelity GP model consistently delivers better accuracy than the single-fidelity GP and neural network models in estimating respiratory-system compliance and resistance ($\sim R^2$). In terms of computational efficiency, our ML model delivers a massive speed-up of $\sim970,000\times$ with respect to the high-fidelity simulations. Regarding lung function, we observed an almost matched and non-linear behavior between some of the structural parameters and chest wall stiffness with compliance. Also, we observed a strong modulation of airways resistance with tissue permeability. Our findings suggest the relevance of certain constitutive lung tissue model parameters and boundary conditions in the respiratory-system response and airway resistance. Furthermore, we highlight the advantages of adopting a multi-fidelity ML approach that combines data from different levels to obtain accurate and efficient estimates in biomechanical simulations. We expect that the methods and results shown here will be valuable for generating lung models and similar organs to support relevant clinical situations.

## Directories
- `results-data`: Figures and related results from processed from `raw-data` and `tests`.
- `raw-data`: Input data needed for simulations + data generated from `tests`.
- `src`: Files that implement functions used in simulations.
- `tests`:  Code files that implement the described work.

## Results
The main results of this research are shown below. Figure 3 shows the airway pressure, flow and volume signals predicted by high and low fidelity models of the lung during PCV simulations.
![fig01_CT_meshes](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/411613eb-f2cb-4a3b-ace9-737cc58b4318)

Figure 1. Construction of high-fidelity and low-fidelity lung finite-element models. (a) Computed-tomography image from which the lung domain is determined, (b) high-fidelity finite-element mesh generated from image lung domain, and (c) finite-element mesh generated for the low-fidelity lung model.



![fig02_HFvsLF](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/bed69591-09de-4d43-b995-d2e176862221)

Figure 2. Simulation of lungs under PCV ventilation model. Physiological signals that describe the time evolution of (a) airways pressure, (b) flow, and (c) volume (c) are shown for the high-fidelity (solid lines) and low-fidelity model (dashed lines).



![fig03_rmse](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/efe4e193-53e4-449c-8ca3-acaef6949e3b)

Figure 3. Effect of the equivalent high-fidelity cost (training sample size) on the prediction performance of the multi-fidelity GP, single-fidelity GP, and neural network models. (a) Respiratory-system compliance, and (b) Resistance. Results shown are for the testing set predictions. Dashed lines and error bars denote the average and standard deviation of the RMSE, respectively.



![fig4_blandaltman_c](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/6f299abb-3ebc-4660-b2e6-ec658a518efc)

Figure 4. Performance comparison of multi-fidelity GP, single-fidelity GP, and neural network models on the respiratory-system compliance. All units are in ml/cm H$_{\text{2}}$O. Regarding the high-fidelity data for training, each model was trained with a training size of 95\% (19 observations). The predictions on the testing set are analyzed using (a) correlation plots, and (b) Bland-Altman plots. In Bland-Altman plots, solid lines represent the mean difference between the high-fidelity simulations and model predictions, while dashed lines represent their corresponding $\pm$1.96 standard deviations.



![fig05_blandaltman_r](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/22612436-47c9-4dcb-b609-6c7acbd1c3a9)

Figure 5. Performance comparison of multi-fidelity GP, single-fidelity GP and neural network models on the airways resistance. All units are in cm H$_{\text{2}}$O/L/s. Regarding the high-fidelity data for training, each model was trained with a training size of 95\% (19 observations). The predictions on the testing set are analyzed using (a) correlation plots, and (b) Bland-Altman plots. In Bland-Altman plots, solid lines represent the mean difference between the high-fidelity simulations and model predictions, while dashed lines represent their corresponding $\pm$1.96 standard deviations.



![fig06_sobol_indices](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/5909b9dc-37fa-4cd5-94a0-c1079c62b85c)

Figure 6. Sobol total sensitivity indices with respect to lung model parameters [$c$, $\beta$, $c_1$, $c_3$, $k$, $K_s$], for (a) respiratory-system compliance, and (b) airways resistance. Each bar indicates the total-order index, while error bars indicate the 95\% confidence intervals.



![fig07_maineffects_c](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/32b43c32-805e-4bdb-a5ff-d486de7390e6)

Figure 7. Main effects analysis of the respiratory-system compliance. Parameters $c$, $\beta$ and $K_s$ show an inverse relationship with compliance. Furthermore, $c$ and $K_s$ have the greatest influence on the response. On the other hand, $k$ shows a slightly direct relation with compliance. Parameters $c_1$ and $c_3$ do not influence the response of compliance. Gray lines represent the 100 trajectories used for the simulation, while black lines represent the average main effect.


![fig08_maineffects_r](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/ce7821cc-2959-4980-84f9-9a613f4cb706)

Figure 8. Main effects analysis of the airways resistance $\text{R}$. The permeability parameter $k$ is the only one with influence on the response. It can be observed that this parameter shows a non-linear inverse relationship with resistance. Gray lines represent the 100 trajectories used for the simulation, while black lines represent the average main effect.



![fig09_uncertainty_example](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/f2138c31-ce97-4933-bcfa-1f3c1da47187)

Figure 9. Uncertainty propagation of constitutive model parameter $c$ and its effect on the response variability. This example case considers a $\pm$25\% uncertainty with respect to the baseline value, where (a) corresponds to the uniform probability distribution of parameter $c$, which is applied to our surrogate model, obtaining in (b) empirical distributions for respiratory-system-compliance and resistance response. The dashed vertical line represents the baseline value for $c$ and the, while solid vertical lines represent the reference response of the model for the parameters baseline values. Solid curves represent the corresponding probability density functions.



![fig10_parameter_uncertainty](https://github.com/comp-medicine-uc/ML-lung-mechanics-UQ/assets/95642663/1e0dd154-508a-4ad6-9908-093a645a51fc)

Figure 10. Uncertainty propagation analysis for respiratory-system compliance and airways resistance. Three levels of variability in the input probability distribution are considered: $\Delta = \pm10\%, \pm25\%, \pm50\%$. Each row corresponds to one of the six parameters [$c$, $\beta$, $c_1$, $c_3$, $k$, $K_s$]. The last row shows the variability when uncertainty is present in all parameters simultaneously (All). Vertical lines correspond to values obtained for the baseline parameters.

## Dependencies
- `FeniCS` 2019.1.0
- `numpy`
- `scipy`
- `matplotlib`
- `GPy`
- `Emukit`
- `os`
- `sys`

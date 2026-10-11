<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/9feb7f4a-a005-400a-a284-eebc4da6dda5" /># Qiskit Fall Fest 2026 — Project Submission Template
**IIIT Kottayam Quantum Hackathon (24-Hour Sprint)**

> ⚠️ **CRITICAL INSTRUCTIONS FOR PARTICIPANTS:**
> 1. **Do NOT rename or delete the section headings** below. The automated AI evaluation and filtering system relies on this exact Markdown structure to parse and grade your project.
> 2. Replace all placeholder text enclosed in `[brackets]` with your project's details.
> 3. **Commit History Audit:** All project commits must be timestamped strictly within the 24-hour hackathon window (between **October 10, 2026 at 7:05 PM** and **October 11, 2026 at 7:05 PM**). Commits observed outside this window will result in **immediate disqualification**.
> 4. Ensure all Jupyter Notebooks have their cells **run and outputs displayed** before pushing.
> 5. Starter resources and reference guides are located in the [`references/`](./references/) folder.
> 6. **Tech Stack Flexibility:** You are free to use any external frameworks/technologies (e.g., Streamlit, React, Pygame, FastAPI, NetworkX) for your frontend, app logic, or classical baselines. However, **Qiskit MUST power the quantum core** of your project.
> 7. **Full Event Guide & Rules:** For full schedule details, track inspirations, submission steps, and grading focus areas, refer to [`HACKATHON_GUIDE.md`](./HACKATHON_GUIDE.md).

*(You may delete the instructions box above when preparing your final repository submission)*

---

# Project Title

## Project Metadata

* **Project Title:** `[“Engine Performance Prediction Using Cubic Spline Interpolation and Quantum Machine Learning: A Comparative Study of Variational Quantum Regression and Classical Quadratic Regression”]`
* **Team Name:** `[“Quantum Kernel Machine Learning for Data-Driven Analysis of Petrol Internal Combustion Engine Performance Characteristics”]`
* **Chosen Track:** `[ Track 2: Open-Domain Optimization & Real-World Solvers]`
* **Video Presentation Link:** `[Insert YouTube Unlisted / Google Drive / Loom URL here]`
  *(⚠️ Crucial: Ensure link access is set to "Anyone with the link can view" on Google Drive or "Unlisted / Public" on YouTube. Locked/inaccessible links cannot be scored!)*
* **Live Demo Timestamp in Video:** `[MM:SS]` *(e.g., "Demo starts at 03:45")*

### Team Members (Max 4 members)
| Name | Email | GitHub Username |
| :--- | :--- | :--- |
| `[Mahendrakar Balaji rao]` | `[balajirao304@gmail.com]` | `[@balajirao304-ship-it]` |
| `[Chukkala kiran]` | `[ckiranme@gmail.com]` | `[@chukkala-kiran]` |
| `[Mendu vinay]` | `[vinaymendu18@gmail.com]` | `[@Menduvinay2003]` |
| `[S Manikanta]` | `[manikantasuper63@gmail.com]` | `[@manikantasuper63-hash]` |

---

## 1. Project Overview & Motivation

* **Summary (3–4 sentences):**  
  `[This project develops an engine performance prediction model using cubic spline interpolation, classical quadratic regression, and Variational Quantum Regression (VQR) implemented with Qiskit. It analyzes five performance parameters: fuel consumption, specific fuel consumption, brake thermal efficiency, indicated thermal efficiency, and mechanical efficiency. The main optimization dilemma is to achieve accurate predictions while minimizing model error and balancing computational complexity between classical and quantum approaches. The models are compared using \(R^2\), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to evaluate their predictive performance.]`

* **Why this problem / game?**  
  `[This topic was chosen to explore how advanced computational techniques can improve engine performance analysis and prediction. Accurate prediction of fuel consumption, specific fuel consumption, and thermal efficiency is important for evaluating engine performance and identifying opportunities to improve fuel economy. The project is engaging because it combines mechanical engineering, data interpolation, classical machine learning, and quantum computing using Qiskit. Comparing Variational Quantum Regression with classical quadratic regression provides valuable practical learning about model accuracy, optimization, and the potential applications and limitations of quantum machine learning in engineering]`

---

## 2. Quantum Architecture & Mechanics

* **Core Quantum Principles Used:**  
  `[

1. **Superposition:** Qubits represent combinations of quantum states through parameterized rotation gates in the variational circuit.
2. **Entanglement:** The ZZFeatureMap uses controlled interactions between qubits to encode relationships between input features.
3. **Quantum Interference:** Quantum gate operations combine probability amplitudes, influencing the circuit's output and its ability to represent nonlinear relationships.
4. **Measurement:** The quantum circuit's output is evaluated through a quantum observable or measurement process to obtain values used for prediction.
5. **Variational Optimization:** Trainable quantum gates are optimized using the classical COBYLA optimizer to minimize the prediction error between actual and predicted engine performance values.

These principles are applied in Variational Quantum Regression (VQR) using Qiskit to model and compare five engine performance curves against classical quadratic regression. The project does not use QAOA because it addresses a regression problem rather than a combinatorial optimization problem.]`

* **How the Quantum Circuit is Integrated:**  
  `[**How the Quantum Circuit is Integrated:**

The Qiskit quantum circuit is integrated into the engine performance prediction system using Variational Quantum Regression (VQR). The input variable, brake power, is normalized and encoded into quantum states using a two-qubit ZZFeatureMap, while nonlinear features help represent relationships between brake power and engine performance parameters. A trainable RealAmplitudes ansatz processes the encoded inputs, and a classical COBYLA optimizer adjusts the quantum circuit parameters to minimize the prediction error. The resulting VQR predictions for fuel consumption, specific fuel consumption, brake thermal efficiency, indicated thermal efficiency, and mechanical efficiency are compared with classical quadratic regression using R², MAE, and RMSE metrics.]`

* **Circuit Details:**  
  * **Number of Qubits:** `[ 2 qubits]`
  * **Key Gates Used:** `[ZZFeatureMap entangling gates, parameterized rotation gates (RY), and controlled quantum interactions. The RealAmplitudes ansatz provides trainable rotation and entanglement operations]`
  * **Primitives Used:** `[StatevectorEstimator for evaluating quantum observables during model training and prediction.]`
  * **Other Technologies Used:** `[Python, Qiskit, Qiskit Machine Learning, NumPy, Pandas, SciPy (cubic spline and PCHIP interpolation), Scikit-learn (classical regression and evaluation metrics), and Matplotlib (performance curves and comparison graphs)]`

```python
# Optional: Paste a representative snippet of your core Qiskit circuit code here
!pip install -q qiskit qiskit-aer

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# =====================================
# Engine experimental data
# =====================================

voltage = 200
current = np.array([0, 1.5, 2.5, 3.0, 4.0,
                    4.5, 6.0, 7.0, 7.5])

time = np.array([39.34, 38.75, 38.35, 33.24,
                 29.20, 29.20, 20.90, 21.07, 16.70])

# Brake power in kW
brake_power = voltage * current / 1000

# =====================================
# Select one operating point
# =====================================

index = 5
bp = brake_power[index]

# Normalize brake power to [0, 1]
bp_norm = bp / brake_power.max()

# Convert normalized value to a rotation angle
theta = np.pi * bp_norm

print("Brake Power:", bp, "kW")
print("Normalized BP:", bp_norm)
print("Rotation angle:", theta, "radians")

# =====================================
# Create a one-qubit quantum circuit
# =====================================

qc = QuantumCircuit(1, 1)

# Encode brake power using a Y rotation
qc.ry(theta, 0)

# Measure the qubit
qc.measure(0, 0)

print("\nQuantum circuit:")
print(qc.draw(output="text"))

# =====================================
# Simulate the circuit
# =====================================

simulator = AerSimulator()

compiled_circuit = transpile(qc, simulator)

result = simulator.run(
    compiled_circuit,
    shots=2048
).result()

counts = result.get_counts()

print("\nMeasurement counts:")
print(counts)

# =====================================
# Plot measurement probabilities
# =====================================

total_shots = sum(counts.values())

p0 = counts.get("0", 0) / total_shots
p1 = counts.get("1", 0) / total_shots

print("Probability of 0:", p0)
print("Probability of 1:", p1)

plt.bar(["0", "1"], [p0, p1])
plt.xlabel("Measured qubit state")
plt.ylabel("Probability")
plt.title("Qiskit Circuit: Brake Power Encoding")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show()
```

---

## 3. Repository Structure & How to Run

### Project Structure
```text
├── [README.md  ](https://github.com/balajirao304-ship-it/sturdy-garbanzo/tree/main)             # Project documentation (this file)
├── requirements.txt        # Python dependencies
├── src/                    # Source code scripts or game application
│   └── main.py             # Main entry point (or streamlit app)
├── notebooks/              # Jupyter notebooks containing experiments
│   └── performance_of_petrol_engine_by_qml.ipynb      # Main runnable notebook with outputs
└── references/             # Hackathon reference guides (Track 1 & Track 2)
```

### Installation & Execution Instructions

> 💡 **Environment Recommendation:** To avoid package conflicts, we strongly suggest using a fresh Python virtual environment or Google Colab:
> ```bash
> python -m venv venv
> source venv/bin/activate    # On Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

1. **Clone the repository:**
   ```bash
   git clone [your-repo-url]
   cd [your-repo-folder]
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the project / demo:**
   ```bash
   # If running a script or CLI game:
   python src/main.py

   # If running a Streamlit web app:
   streamlit run src/app.py

   # Or open the Jupyter Notebook:
   jupyter notebook notebooks/solution.ipynb
   ```

---

## 4. Implementation Walkthrough & Live Demo

* **Walkthrough of Gameplay / Optimization Pipeline:**  
  `[Describe the user experience or algorithmic workflow step-by-step. What inputs does the user provide, what happens inside the quantum circuit, and what output is produced?]`

* **Key Screenshots / Terminal Output:**  
  `[Paste sample terminal output, game screenshots, or optimization convergence plots here to demonstrate that the project runs.]`

---

## 5. Classical vs. Quantum Comparison

* **Classical Baseline / Counterpart:**  
  `[What classical baseline did you compare against? (e.g., a classical player, a random guesser, a brute-force classical solver, or a naive greedy heuristic)]`

* **Comparative Analysis:**  
  `[### Comparative Analysis

The Variational Quantum Regression (VQR) model implemented using Qiskit is compared with Classical Quadratic Regression for predicting five engine performance parameters: fuel consumption, specific fuel consumption, brake thermal efficiency, indicated thermal efficiency, and mechanical efficiency.

Classical Quadratic Regression uses a second-degree polynomial, while VQR uses a two-qubit quantum feature map and a trainable variational circuit optimized using COBYLA. Both models are evaluated on the same held-out experimental observations using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

The comparison examines prediction accuracy, model stability, and computational requirements. VQR may capture nonlinear relationships through quantum feature encoding, but its performance depends on the dataset, circuit configuration, and optimizer settings. Classical quadratic regression provides a simpler and computationally inexpensive baseline.

Since this project focuses on engine performance prediction rather than a combinatorial optimization problem, the Track 2 approximation ratio and feasibility yield are not applicable. No quantum advantage is claimed without actual experimental results. The final conclusion will be based on the measured evaluation metrics, with the small dataset acknowledged as a limitation.]`

---

## 6. Performance, Results & Limitations

* **Key Results & Observations:**  
  `[    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.8/9.8 MB 124.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.4/12.4 MB 126.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.4/2.4 MB 111.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.9/54.9 kB 7.1 MB/s eta 0:00:00
Brake Power: 0.9 kW
Normalized BP: 0.6
Rotation angle: 1.8849555921538759 radians

Quantum circuit:
     ┌──────────┐┌─┐
  q: ┤ Ry(3π/5) ├┤M├
     └──────────┘└╥┘
c: 1/═════════════╩═
                  0 

Measurement counts:
{'0': 714, '1': 1334}
Probability of 0: 0.3486328125
Probability of 1: 0.6513671875
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/8038e956-af8d-400a-8026-3cb4ef7c6224" />
https://colab.research.google.com/drive/1yy7oqdXUg4YX89qaQNomf44JhLJH1E8M#scrollTo=L-g2ZDWPkPUi&line=1&uniqifier=1
 
ORIGINAL PERFORMANCE DATA
 Brake Power  Fuel Consumption    SFC  Brake Thermal Efficiency  Indicated Thermal Efficiency  Mechanical Efficiency
         0.3            0.6875 2.2916                    3.5703                        9.1639                38.9610
         0.5            0.6947 1.3893                    5.8891                       11.4249                51.5464
         0.6            0.8014 1.3357                    6.1253                       10.9235                56.0748
         0.8            0.9123 1.1404                    7.1744                       11.3894                62.9921
         0.9            0.9123 1.0137                    8.0713                       12.2862                65.6934
         1.2            1.2746 1.0622                    7.7027                       10.7196                71.8563
         1.4            1.2644 0.9031                    9.0596                       12.1010                74.8663
         1.5            1.5952 1.0635                    7.6935                       10.1041                76.1421

Training observations: 5
Test observations: 3

============================================================
PERFORMANCE PARAMETER: Fuel Consumption
============================================================
Classical equation: y = 1.164861 + (-1.075515) BP + (0.872764) BP^2
/tmp/ipykernel_1549/2632585065.py:225: DeprecationWarning: The class ``qiskit.circuit.library.data_preparation._zz_feature_map.ZZFeatureMap`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the zz_feature_map function as a replacement. Note that this will no longer return a BlueprintCircuit, but just a plain QuantumCircuit.
  feature_map = ZZFeatureMap(
/tmp/ipykernel_1549/2632585065.py:230: DeprecationWarning: The class ``qiskit.circuit.library.n_local.real_amplitudes.RealAmplitudes`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the function qiskit.circuit.library.real_amplitudes instead.
  ansatz = RealAmplitudes(
Classical Quadratic    | R2 = 0.5696 | MAE = 0.1758 | RMSE = 0.1805
Quantum VQR            | R2 = -0.6877 | MAE = 0.2847 | RMSE = 0.3574

============================================================
PERFORMANCE PARAMETER: SFC
============================================================
Classical equation: y = 2.711430 + (-3.016400) BP + (1.259894) BP^2
/tmp/ipykernel_1549/2632585065.py:225: DeprecationWarning: The class ``qiskit.circuit.library.data_preparation._zz_feature_map.ZZFeatureMap`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the zz_feature_map function as a replacement. Note that this will no longer return a BlueprintCircuit, but just a plain QuantumCircuit.
  feature_map = ZZFeatureMap(
/tmp/ipykernel_1549/2632585065.py:230: DeprecationWarning: The class ``qiskit.circuit.library.n_local.real_amplitudes.RealAmplitudes`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the function qiskit.circuit.library.real_amplitudes instead.
  ansatz = RealAmplitudes(
Classical Quadratic    | R2 = 0.7790 | MAE = 0.2189 | RMSE = 0.2444
Quantum VQR            | R2 = -0.9425 | MAE = 0.5332 | RMSE = 0.7246

============================================================
PERFORMANCE PARAMETER: Brake Thermal Efficiency
============================================================
Classical equation: y = -2.509039 + (18.731552) BP + (-7.763255) BP^2
/tmp/ipykernel_1549/2632585065.py:225: DeprecationWarning: The class ``qiskit.circuit.library.data_preparation._zz_feature_map.ZZFeatureMap`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the zz_feature_map function as a replacement. Note that this will no longer return a BlueprintCircuit, but just a plain QuantumCircuit.
  feature_map = ZZFeatureMap(
/tmp/ipykernel_1549/2632585065.py:230: DeprecationWarning: The class ``qiskit.circuit.library.n_local.real_amplitudes.RealAmplitudes`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the function qiskit.circuit.library.real_amplitudes instead.
  ansatz = RealAmplitudes(
Classical Quadratic    | R2 = 0.5955 | MAE = 1.0729 | RMSE = 1.0757
Quantum VQR            | R2 = -1.8450 | MAE = 2.4225 | RMSE = 2.8526

============================================================
PERFORMANCE PARAMETER: Indicated Thermal Efficiency
============================================================
Classical equation: y = 2.810455 + (18.310563) BP + (-8.714773) BP^2
/tmp/ipykernel_1549/2632585065.py:225: DeprecationWarning: The class ``qiskit.circuit.library.data_preparation._zz_feature_map.ZZFeatureMap`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the zz_feature_map function as a replacement. Note that this will no longer return a BlueprintCircuit, but just a plain QuantumCircuit.
  feature_map = ZZFeatureMap(
/tmp/ipykernel_1549/2632585065.py:230: DeprecationWarning: The class ``qiskit.circuit.library.n_local.real_amplitudes.RealAmplitudes`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the function qiskit.circuit.library.real_amplitudes instead.
  ansatz = RealAmplitudes(
Classical Quadratic    | R2 = -1.8692 | MAE = 1.5989 | RMSE = 1.6000
Quantum VQR            | R2 = -1.7229 | MAE = 1.1362 | RMSE = 1.5587

============================================================
PERFORMANCE PARAMETER: Mechanical Efficiency
============================================================
Classical equation: y = 28.197391 + (56.416338) BP + (-16.375939) BP^2
/tmp/ipykernel_1549/2632585065.py:225: DeprecationWarning: The class ``qiskit.circuit.library.data_preparation._zz_feature_map.ZZFeatureMap`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the zz_feature_map function as a replacement. Note that this will no longer return a BlueprintCircuit, but just a plain QuantumCircuit.
  feature_map = ZZFeatureMap(
/tmp/ipykernel_1549/2632585065.py:230: DeprecationWarning: The class ``qiskit.circuit.library.n_local.real_amplitudes.RealAmplitudes`` is deprecated as of Qiskit 2.1. It will be removed in Qiskit 3.0. Use the function qiskit.circuit.library.real_amplitudes instead.
  ansatz = RealAmplitudes(
Classical Quadratic    | R2 = 0.9587 | MAE = 1.9707 | RMSE = 2.7549
Quantum VQR            | R2 = -1.3684 | MAE = 20.4436 | RMSE = 20.8566

FINAL TEST-SET COMPARISON
                   Parameter               Model  Test R2  Test MAE  Test RMSE
            Fuel Consumption Classical Quadratic   0.5696    0.1758     0.1805
            Fuel Consumption         Quantum VQR  -0.6877    0.2847     0.3574
                         SFC Classical Quadratic   0.7790    0.2189     0.2444
                         SFC         Quantum VQR  -0.9425    0.5332     0.7246
    Brake Thermal Efficiency Classical Quadratic   0.5955    1.0729     1.0757
    Brake Thermal Efficiency         Quantum VQR  -1.8450    2.4225     2.8526
Indicated Thermal Efficiency Classical Quadratic  -1.8692    1.5989     1.6000
Indicated Thermal Efficiency         Quantum VQR  -1.7229    1.1362     1.5587
       Mechanical Efficiency Classical Quadratic   0.9587    1.9707     2.7549
       Mechanical Efficiency         Quantum VQR  -1.3684   20.4436    20.8566

PREDICTIONS: Fuel Consumption
 Brake Power (kW)  Actual  Classical Prediction  Quantum VQR Prediction
              0.5  0.6947                0.8453                  1.2818
              1.2  1.2746                1.1310                  1.1036
              0.3  0.6875                0.9208                  0.7833

PREDICTIONS: SFC
 Brake Power (kW)  Actual  Classical Prediction  Quantum VQR Prediction
              0.5  1.3893                1.5182                  1.0872
              1.2  1.0622                0.9060                  1.1444
              0.3  2.2916                1.9199                  1.0763

PREDICTIONS: Brake Thermal Efficiency
 Brake Power (kW)  Actual  Classical Prediction  Quantum VQR Prediction
              0.5  5.8891                4.9159                  7.7253
              1.2  7.7027                8.7897                  6.7605
              0.3  3.5703                2.4117                  8.0595

PREDICTIONS: Indicated Thermal Efficiency
 Brake Power (kW)  Actual  Classical Prediction  Quantum VQR Prediction
              0.5 11.4249                9.7870                 10.8235
              1.2 10.7196               12.2339                 10.9012
              0.3  9.1639                7.5193                 11.7894

PREDICTIONS: Mechanical Efficiency
 Brake Power (kW)  Actual  Classical Prediction  Quantum VQR Prediction
              0.5 51.5464               52.3116                 74.6175
              1.2 71.8563               72.3156                 57.2442
              0.3 38.9610               43.6485                 62.6086

Saved:
1. engine_quantum_classical_metrics.csv
2. engine_performance_interpolation_vqr.png
.]`

* **Critical Reflection on Limitations:**  
  `[### Critical Reflection on Limitations

Although this project demonstrates the application of Variational Quantum Regression (VQR) to engine performance prediction, several limitations must be considered.

1. **Limited experimental dataset:** The model uses only a small number of engine observations. Consequently, predictions may be sensitive to the train/test split, and the evaluation metrics may not reliably represent performance on unseen data. Collecting more experimental measurements and applying cross-validation would improve reliability.

2. **Scalability with larger qubit counts:** Increasing the number of qubits can increase the computational cost of quantum circuit simulation. Classical statevector simulation typically requires memory that grows exponentially with the number of qubits. Larger circuits may also require more trainable parameters and optimization iterations, making training more challenging.

3. **Quantum noise and gate errors:** Real quantum processors contain imperfect quantum gates and other sources of noise. These errors can distort the quantum states prepared by the feature map and variational circuit, reducing the accuracy and consistency of predicted engine performance parameters.

4. **Decoherence:** Qubits can lose their quantum properties through interactions with their environment. Longer circuits and deeper gate sequences increase exposure to decoherence, potentially reducing the reliability of the computation on noisy intermediate-scale quantum (NISQ) hardware.

5. **Optimization difficulties:** The COBYLA optimizer may converge to suboptimal parameter values. Variational quantum models can be sensitive to parameter initialization, circuit architecture, and optimization settings. Multiple initializations and repeated experiments are needed to assess stability.

6. **No demonstrated quantum advantage:** The two-qubit VQR model is evaluated using ideal statevector-based simulation, which does not reproduce all the noise and operational constraints of real quantum hardware. Any improvement over classical regression must be demonstrated through measured results and robust validation. Quantum computing is not automatically more accurate or faster than classical methods for small regression problems.

7. **Interpolation limitations:** Cubic spline interpolation generates smooth curves between measured points, but these curves do not establish physical correctness or provide independent experimental evidence. Predictions outside the measured range may be particularly unreliable.

8. **Engineering data limitations:** The accuracy of the predictions depends on the quality of the experimental measurements and the validity of the performance calculations. In particular, electrical power calculated as voltage multiplied by current should not automatically be interpreted as engine brake power unless justified by the experimental setup.

**Future improvements:** Future work should include collecting a larger engine dataset, using repeated cross-validation, testing additional classical regression models, investigating deeper or alternative quantum circuits, and evaluating VQR on noisy quantum simulators or physical quantum hardware. These steps would help establish the model's reliability, scalability, and practical usefulness in engineering applications.]`

---

## 7. References & Acknowledgments

* `[### References

1. Qiskit Documentation. *Qiskit: An Open-Source Quantum Computing SDK.* https://docs.quantum.ibm.com/
2. Qiskit Machine Learning Documentation. *Quantum machine learning algorithms and model implementations.* https://qiskit-community.github.io/qiskit-machine-learning/
3. SciPy Documentation. *Interpolation and Cubic Spline Functions.* https://docs.scipy.org/doc/scipy/reference/interpolate.html
4. Scikit-learn Documentation. *Polynomial Regression, Model Evaluation, and Regression Metrics.* https://scikit-learn.org/stable/
5. NumPy Documentation. https://numpy.org/doc/
6. Matplotlib Documentation. https://matplotlib.org/stable/

### Acknowledgments

This project uses open-source Python libraries for numerical calculations, interpolation, regression, quantum circuit construction, optimization, and visualization. Qiskit provides the quantum machine learning framework, while SciPy and scikit-learn support the interpolation and classical comparison methods.

**Team contributions:** Member 1 - M.Balaji Rao – Data Collection and Engine Performance Calculations
Member 2 - chukkal kiran – Classical Regression and Interpolation
Member 3 – Mendu vinay - Quantum Machine Learning Implementation
Member 4 –  s. Manikanta - Model Evaluation, Documentation, and Presentation


**AI assistance:** If applicable, acknowledge the use of AI tools for coding assistance, documentation, or debugging in accordance with the hackathon's rules..]`
chatGpt
* `[### Acknowledgment of Team Contributions and Tools Used

We sincerely acknowledge the contributions of all four team members in completing this project, **“Engine Performance Prediction Using Cubic Spline Interpolation and Quantum Machine Learning: A Comparative Study of Variational Quantum Regression and Classical Quadratic Regression.”** The team collaborated on experimental data preparation, engine performance calculations, classical regression, quantum circuit implementation, model evaluation, visualization, documentation, and presentation.

The following tools and technologies were used during the project:

- **Python:** Programming and implementation.
- **Qiskit:** Quantum circuit construction and quantum machine learning.
- **Qiskit Machine Learning:** Implementation of Variational Quantum Regression (VQR).
- **NumPy and Pandas:** Numerical computations and data processing.
- **SciPy:** Cubic spline interpolation and optimization support.
- **Scikit-learn:** Classical quadratic regression and model evaluation.
- **Matplotlib:** Data visualization and comparison graphs.
- **Jupyter Notebook / Google Colab:** Code execution and experimentation.
- **ChatGPT:** Assistance with coding guidance, debugging suggestions, explanations of quantum machine learning concepts, and project documentation.

We thank the developers and open-source communities behind these tools for making them available for educational and research purposes. All team members share responsibility for reviewing the implementation, validating the experimental results, and ensuring that the final submission follows the hackathon guidelines..]`

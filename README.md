# Qiskit Fall Fest 2026 — Project Submission Template
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
```

---

## 3. Repository Structure & How to Run

### Project Structure
```text
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python dependencies
├── src/                    # Source code scripts or game application
│   └── main.py             # Main entry point (or streamlit app)
├── notebooks/              # Jupyter notebooks containing experiments
│   └── solution.ipynb      # Main runnable notebook with outputs
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
  `[How did the quantum approach compare to the classical baseline? For Track 1: Did quantum strategy give an unfair advantage or unique gameplay behavior? For Track 2: What was the approximation ratio (C_quantum / C_optimal) and feasibility yield?]`

---

## 6. Performance, Results & Limitations

* **Key Results & Observations:**  
  `[Summarize the main results from your experiments, simulations, or gameplay tests.]`

* **Critical Reflection on Limitations:**  
  `[Be honest and rigorous: What happens if this is scaled to larger qubit counts? How would physical quantum noise, gate errors, and decoherence affect your solution on real NISQ hardware? Acknowledging limitations demonstrates true scientific understanding!]`

---

## 7. References & Acknowledgments

* `[Cite any open-source libraries, GitHub repositories, or research papers you referenced during the hackathon.]`
* `[Acknowledge team contributions and tools used.]`

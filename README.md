# Greenwashing-Project
Code for Monte Carlo simulations and machine learning analysis used in EXPLAINABLE MACHINE LEARNING PREDICTIONS FOR PREVENTING GREENWASHING: AN EVOLUTIONARY GAME-THEORETIC ANALYSIS
This repository contains Python code for a Monte Carlo simulation and machine learning analysis used to model and predict strategic interactions among governments, companies, and the public in the context of greenwashing governance.

Contents
Machine Learning.py: Main script implementing:
Monte Carlo simulations to evaluate strategy payoffs.
Dynamic strategy adjustment probabilities.
Machine learning models to classify strategies.
SHapley Additive exPlanations (SHAP) for interoperability.

Features
Monte Carlo Simulation:
Simulates interactions among three agents: governments, companies, and the public.
Evaluates payoffs based on varying parameters (e.g., costs, benefits, reputation effects).
Includes multiple parameter sets for robust testing.

Machine Learning Analysis:
Trains Random Forest models to classify strategies for each agent.
Optimizes models using grid search for hyperparameter tuning.
Balances imbalanced datasets through resampling.

SHAP Analysis:
Generates SHAP plots to explain the influence of features on strategy predictions.
Visualizes the importance of factors like penalties, monitoring costs, and reputation impacts.

Requirements:
Python 3.8+
Required libraries:
numpy
pandas
matplotlib
sklearn
shap
Install dependencies using: pip install -r requirements.txt

Key Parameters
The following parameters are sampled and used in simulations:
Government: Regulatory costs, fines, cultural guidance costs, reputation gains.
Companies: Greenwashing costs, benefits, reputation losses.
Public: Monitoring costs, loss/benefit effects, reputation impacts.

Data Structure
Input Features:
Variables representing costs, benefits, and probabilities for all agents (e.g., fine, greenwashing_cost, monitoring_cost).
Outputs:
Probabilities of strategy selection (A1, A2, A3 for governments; B1, B2, B3 for companies; C1, C2, C3 for the public).
Payoff outcomes for each agent.

SHAP Visualizations
SHAP summary and bar plots explain feature contributions to the predicted strategies:
Government: Government_A1_summary_plot.png
Company: Company_B1_summary_plot.png
Public: Public_C1_summary_plot.png

License
This code is licensed under the MIT License. 

Contact
For questions or collaboration, please contact Chu Xu at xuchu19thu@gmail.com.

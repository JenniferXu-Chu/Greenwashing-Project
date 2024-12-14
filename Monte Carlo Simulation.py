import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import shap
import os
import scipy.io as scio
from scipy.io import savemat

np.random.seed(42)

government_strategies = ["A1", "A2", "A3"]
company_strategies = ["B1", "B2", "B3"]
public_strategies = ["C1", "C2", "C3"]



def sample_parameters():
    government_cost = np.random.uniform(0, 150)
    fine = np.random.normal(150, 30)
    reputation_gain = np.random.uniform(0, 150)
    cultural_cost = np.random.uniform(0, 150)
    new_reputation_gain = np.random.uniform(0, 150)

    greenwashing_cost = np.random.uniform(0, 150)
    green_benefit = np.random.normal(150, 30)
    reputation_loss = np.random.normal(80, 10)
    passive_greenwashing_benefit = np.random.normal(80, 30)

    monitoring_cost = np.random.uniform(0, 150)
    public_loss_benefit = np.random.normal(150, 30)
    public_reputation_loss = np.random.normal(80, 10)
    public_reputation_gain = np.random.uniform(0, 150)
    public_new_reputation_gain = np.random.uniform(0, 150)
    passive_greenwashing_loss = np.random.normal(80, 30)
    discovery_prob_single = np.random.uniform(0, 0.5)
    discovery_prob_both = np.random.uniform(discovery_prob_single, 1)

    return {
        "government_cost": government_cost,
        "fine": fine,
        "reputation_gain": reputation_gain,
        "cultural_cost": cultural_cost,
        "new_reputation_gain": new_reputation_gain,
        "greenwashing_cost": greenwashing_cost,
        "green_benefit": green_benefit,
        "public_loss_benefit": public_loss_benefit,
        "reputation_loss": reputation_loss,
        "passive_greenwashing_benefit": passive_greenwashing_benefit,
        "public_reputation_loss": public_reputation_loss,
        "public_reputation_gain": public_reputation_gain,
        "public_new_reputation_gain": public_new_reputation_gain,
        "passive_greenwashing_loss": passive_greenwashing_loss,
        "monitoring_cost": monitoring_cost,
        "discovery_prob_single": discovery_prob_single,
        "discovery_prob_both": discovery_prob_both
    }



def calculate_payoffs(government_strategy, company_strategy, public_strategy, params):
    if government_strategy == "A1":
        if company_strategy == "B1":
            if public_strategy == "C1":

                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] - \
                                    params["government_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - (params["reputation_loss"] + params["fine"]) * params[
                                     "discovery_prob_single"]
                public_payoff = params["public_reputation_gain"] - (1 - params["discovery_prob_single"]) * params[
                    "public_loss_benefit"] - params["discovery_prob_single"] * params["public_reputation_loss"]
            elif public_strategy == "C2":

                government_payoff = params["discovery_prob_single"] * params["fine"] - params["government_cost"]
                company_payoff = - params["greenwashing_cost"] - params["fine"] * params["discovery_prob_single"]
                public_payoff = 0
            elif public_strategy == "C3":

                government_payoff = params["discovery_prob_both"] * params["fine"] + params["reputation_gain"] - params[
                    "government_cost"]
                company_payoff = (1 - params["discovery_prob_both"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - params["discovery_prob_both"] * (params["reputation_loss"] + params["fine"])
                public_payoff = params["public_reputation_gain"] - (1 - params["discovery_prob_both"]) * params[
                    "public_loss_benefit"] - params["monitoring_cost"] - params["discovery_prob_both"] * params[
                                    "public_reputation_loss"]
        elif company_strategy == "B2":
            if public_strategy == "C1":

                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] - \
                                    params["government_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["passive_greenwashing_benefit"] - \
                                 params["discovery_prob_single"] * params["fine"]
                public_payoff = params["public_reputation_gain"] - (1 - params["discovery_prob_single"]) * params[
                    "passive_greenwashing_loss"]
            elif public_strategy == "C2":

                government_payoff = params["discovery_prob_single"] * params["fine"] - params["government_cost"]
                company_payoff = -params["discovery_prob_single"] * params["fine"]
                public_payoff = 0
            elif public_strategy == "C3":

                government_payoff = params["discovery_prob_both"] * params["fine"] + params["reputation_gain"] - params[
                    "government_cost"]
                company_payoff = (1 - params["discovery_prob_both"]) * params["passive_greenwashing_benefit"] - params[
                    "discovery_prob_both"] * params["fine"]
                public_payoff = params["public_reputation_gain"] - (1 - params["discovery_prob_both"]) * params[
                    "passive_greenwashing_loss"] - params["monitoring_cost"]
        elif company_strategy == "B3":
            if public_strategy == "C1":

                government_payoff = params["reputation_gain"] - params["government_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"]
            elif public_strategy == "C2":

                government_payoff = - params["government_cost"]
                company_payoff = 0
                public_payoff = 0
            elif public_strategy == "C3":

                government_payoff = params["reputation_gain"] - params["government_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"] - params["monitoring_cost"]

    elif government_strategy == "A2":
        if company_strategy == "B1":
            if public_strategy == "C1":

                government_payoff = 0
                company_payoff = params["green_benefit"] - params["greenwashing_cost"]
                public_payoff = -params["public_loss_benefit"]
            elif public_strategy == "C2":

                government_payoff = 0
                company_payoff = - params["greenwashing_cost"]
                public_payoff = 0
            elif public_strategy == "C3":

                government_payoff = params["discovery_prob_single"] * params["fine"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - (params["reputation_loss"] + params["fine"]) * params[
                                     "discovery_prob_single"]
                public_payoff = -(1 - params["discovery_prob_single"]) * params["public_loss_benefit"] - params[
                    "monitoring_cost"] - params["discovery_prob_single"] * params["public_reputation_loss"]
        elif company_strategy == "B2":
            if public_strategy == "C1":

                government_payoff = 0
                company_payoff = params["passive_greenwashing_benefit"]
                public_payoff = -params["passive_greenwashing_loss"]
            elif public_strategy == "C2":

                government_payoff = 0
                company_payoff = 0
                public_payoff = 0
            elif public_strategy == "C3":

                government_payoff = params["discovery_prob_single"] * params["fine"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["passive_greenwashing_benefit"] - \
                                 params["discovery_prob_single"] * params["fine"]
                public_payoff = -(1 - params["discovery_prob_single"]) * params["passive_greenwashing_loss"] - params[
                    "monitoring_cost"]
        elif company_strategy == "B3":
            if public_strategy == "C1":

                government_payoff = 0
                company_payoff = 0
                public_payoff = 0
            elif public_strategy == "C2":

                government_payoff = 0
                company_payoff = 0
                public_payoff = 0
            elif public_strategy == "C3":

                government_payoff = 0
                company_payoff = 0
                public_payoff = - params["monitoring_cost"]

    elif government_strategy == "A3":
        if company_strategy == "B1":
            if public_strategy == "C1":

                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] + \
                                    params["new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - (params["reputation_loss"] + params["fine"]) * params[
                                     "discovery_prob_single"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_single"]) * params["public_loss_benefit"] - params[
                                    "discovery_prob_single"] * params["public_reputation_loss"]
            elif public_strategy == "C2":

                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] + \
                                    params["new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - (params["reputation_loss"] + params["fine"]) * params[
                                     "discovery_prob_single"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_single"]) * params["public_loss_benefit"] - params[
                                    "discovery_prob_single"] * params["public_reputation_loss"]
            elif public_strategy == "C3":

                government_payoff = params["discovery_prob_both"] * params["fine"] + params["reputation_gain"] + params[
                    "new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_both"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - params["discovery_prob_both"] * (params["reputation_loss"] + params["fine"])
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_both"]) * params["public_loss_benefit"] - params[
                                    "monitoring_cost"] - params["discovery_prob_both"] * params[
                                    "public_reputation_loss"]
        elif company_strategy == "B2":
            if public_strategy == "C1":

                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] + \
                                    params["new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["passive_greenwashing_benefit"] - \
                                 params["discovery_prob_single"] * params["fine"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_single"]) * params["passive_greenwashing_loss"]
            elif public_strategy == "C2":

                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] + \
                                    params["new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["passive_greenwashing_benefit"] - \
                                 params["discovery_prob_single"] * params["fine"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_single"]) * params["passive_greenwashing_loss"]
            elif public_strategy == "C3":

                government_payoff = params["discovery_prob_both"] * params["fine"] + params["reputation_gain"] + params[
                    "new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_both"]) * params["green_benefit"] - params[
                    "discovery_prob_both"] * params["fine"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_both"]) * params["passive_greenwashing_loss"] - params[
                                    "monitoring_cost"]
        elif company_strategy == "B3":
            if public_strategy == "C1":

                government_payoff = params["reputation_gain"] + params["new_reputation_gain"] - params[
                    "government_cost"] - params["cultural_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"]
            elif public_strategy == "C2":

                government_payoff = params["reputation_gain"] + params["new_reputation_gain"] - params[
                    "government_cost"] - params["cultural_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"]
            elif public_strategy == "C3":

                government_payoff = params["reputation_gain"] + params["new_reputation_gain"] - params[
                    "government_cost"] - params["cultural_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - params[
                    "monitoring_cost"]
    return government_payoff, company_payoff, public_payoff



initial_probs = {
    "government": {"A1": 1/3, "A2": 1/3, "A3": 1/3},
    "company": {"B1": 1/3, "B2": 1/3, "B3": 1/3},
    "public": {"C1": 1/3, "C2": 1/3, "C3": 1/3}
}



def update_strategy_probabilities(current_probs, payoffs):
    updated_probs = {}
    for strategy, payoff in payoffs.items():
        updated_probs[strategy] = max(0, current_probs[strategy] * (1 + payoff))


    total = sum(updated_probs.values())
    if total > 0:
        for strategy in updated_probs:
            updated_probs[strategy] /= total
    else:

        num_strategies = len(updated_probs)
        for strategy in updated_probs:
            updated_probs[strategy] = 1 / num_strategies

    return updated_probs



def monte_carlo_simulation(num_simulations=500):
    results = []

    government_probs = initial_probs["government"].copy()
    company_probs = initial_probs["company"].copy()
    public_probs = initial_probs["public"].copy()

    for _ in range(num_simulations):
        params = sample_parameters()

        government_strategy = np.random.choice(list(government_probs.keys()), p=list(government_probs.values()))
        company_strategy = np.random.choice(list(company_probs.keys()), p=list(company_probs.values()))
        public_strategy = np.random.choice(list(public_probs.keys()), p=list(public_probs.values()))

        government_payoff, company_payoff, public_payoff = calculate_payoffs(
            government_strategy, company_strategy, public_strategy, params
        )

        total_payoff = government_payoff + company_payoff + public_payoff

        government_probs = update_strategy_probabilities(government_probs, {
            "A1": government_payoff if government_strategy == "A1" else 0,
            "A2": government_payoff if government_strategy == "A2" else 0,
            "A3": government_payoff if government_strategy == "A3" else 0
        })

        company_probs = update_strategy_probabilities(company_probs, {
            "B1": company_payoff if company_strategy == "B1" else 0,
            "B2": company_payoff if company_strategy == "B2" else 0,
            "B3": company_payoff if company_strategy == "B3" else 0
        })

        public_probs = update_strategy_probabilities(public_probs, {
            "C1": public_payoff if public_strategy == "C1" else 0,
            "C2": public_payoff if public_strategy == "C2" else 0,
            "C3": public_payoff if public_strategy == "C3" else 0
        })


        results.append({
            "government_strategy": government_strategy,
            "company_strategy": company_strategy,
            "public_strategy": public_strategy,
            "government_payoff": government_payoff,
            "company_payoff": company_payoff,
            "public_payoff": public_payoff,
            "total_payoff": total_payoff,
            "government_strategy_probabilities": government_probs,
            "company_strategy_probabilities": company_probs,
            "public_strategy_probabilities": public_probs,
            **params
        })

    return pd.DataFrame(results)



simulation_results = monte_carlo_simulation(500)
print(simulation_results.head())

output_path = "output_data/"
os.makedirs(output_path, exist_ok=True)



def plot_evolution_path(simulation_results):
    evolution_data = {strategy: [] for strategy in [
        'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']}

    for i in range(1, len(simulation_results) + 1):
        current_slice = simulation_results[:i]
        evolution_data['A1'].append((current_slice['government_strategy'] == 'A1').mean())
        evolution_data['A2'].append((current_slice['government_strategy'] == 'A2').mean())
        evolution_data['A3'].append((current_slice['government_strategy'] == 'A3').mean())
        evolution_data['B1'].append((current_slice['company_strategy'] == 'B1').mean())
        evolution_data['B2'].append((current_slice['company_strategy'] == 'B2').mean())
        evolution_data['B3'].append((current_slice['company_strategy'] == 'B3').mean())
        evolution_data['C1'].append((current_slice['public_strategy'] == 'C1').mean())
        evolution_data['C2'].append((current_slice['public_strategy'] == 'C2').mean())
        evolution_data['C3'].append((current_slice['public_strategy'] == 'C3').mean())

    pd.DataFrame(evolution_data).to_csv(f"{output_path}evolution_path_data.csv")

    plt.figure(figsize=(12, 8))
    for strategy, data in evolution_data.items():
        plt.plot(data, label=strategy)
    plt.xlabel("Simulation Steps")
    plt.ylabel("Strategy Selection Probability")
    plt.title("Evolution Path of Strategies")
    plt.legend()
    plt.show()


plot_evolution_path(simulation_results)



def plot_government_strategy_influence(simulation_results):
    strategy_counts = simulation_results.groupby(
        ['company_strategy', 'public_strategy', 'government_strategy']).size().unstack(fill_value=0)
    strategy_counts.to_csv(f"{output_path}government_strategy_influence.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(strategy_counts, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Public Strategy")
    ax.set_ylabel("Company Strategy")
    ax.set_title("Government Strategy Variation with Company & Public Strategies")
    plt.show()



plot_government_strategy_influence(simulation_results)
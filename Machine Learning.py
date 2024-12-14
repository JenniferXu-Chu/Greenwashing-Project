import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
import shap
import os
from sklearn.model_selection import train_test_split


np.random.seed(42)

government_strategies = ["A1", "A2", "A3"]
company_strategies = ["B1", "B2", "B3"]
public_strategies = ["C1", "C2", "C3"]



def sample_parameters():
    # 政府参数
    government_cost = np.random.uniform(0, 150)  # 监管成本
    fine = np.random.normal(150, 30)  # 罚款力度
    reputation_gain = np.random.uniform(0, 150)  # 声誉收益
    cultural_cost = np.random.uniform(0, 150)  # 文化引导成本
    new_reputation_gain = np.random.uniform(0, 150)  # 新声誉收益

    # 企业参数
    greenwashing_cost = np.random.uniform(0, 150)  # 漂绿成本
    green_benefit = np.random.normal(150, 30)  # 漂绿收益
    reputation_loss = np.random.normal(80, 10)  # 声誉损失
    passive_greenwashing_benefit = np.random.normal(80, 30)  # 被动漂绿收益

    # 公众参数
    monitoring_cost = np.random.uniform(0, 150)  # 监督成本
    public_loss_benefit = np.random.normal(150, 30)  # 公众效用损失
    public_reputation_loss = np.random.normal(80, 10)  # 欺骗带来的效用损失
    public_reputation_gain = np.random.uniform(0, 150)  # 声誉效用
    public_new_reputation_gain = np.random.uniform(0, 150)  # 新声誉效用
    passive_greenwashing_loss = np.random.normal(80, 30)  # 被动漂绿公众损失
    discovery_prob_single = np.random.uniform(0, 0.5)  # 单方监督发现概率
    discovery_prob_both = np.random.uniform(discovery_prob_single, 1)  # 双方监督发现概率

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

def sample_parameters_2():
    # 政府参数
    government_cost = np.random.uniform(0, 50)  # 监管成本
    fine = np.random.normal(300, 30)  # 罚款力度
    reputation_gain = np.random.uniform(0, 500)  # 声誉收益
    cultural_cost = np.random.uniform(0, 50)  # 文化引导成本
    new_reputation_gain = np.random.uniform(0, 500)  # 新声誉收益

    # 企业参数
    greenwashing_cost = np.random.uniform(0, 50)  # 漂绿成本
    green_benefit = np.random.normal(1000, 30)  # 漂绿收益
    reputation_loss = np.random.normal(80, 10)  # 声誉损失
    passive_greenwashing_benefit = np.random.normal(500, 30)  # 被动漂绿收益

    # 公众参数
    monitoring_cost = np.random.uniform(0, 150)  # 监督成本
    public_loss_benefit = np.random.normal(1000, 30)  # 公众效用损失
    public_reputation_loss = np.random.normal(80, 10)  # 欺骗带来的效用损失
    public_reputation_gain = np.random.uniform(0, 500)  # 声誉效用
    public_new_reputation_gain = np.random.uniform(0, 500)  # 新声誉效用
    passive_greenwashing_loss = np.random.normal(500, 30)  # 被动漂绿公众损失
    discovery_prob_single = np.random.uniform(0, 0.3)  # 单方监督发现概率
    discovery_prob_both = np.random.uniform(discovery_prob_single, 1)  # 双方监督发现概率

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

def sample_parameters_3():
    # 政府参数
    government_cost = np.random.uniform(0, 300)  # 监管成本
    fine = np.random.normal(100, 30)  # 罚款力度
    reputation_gain = np.random.uniform(0, 100)  # 声誉收益
    cultural_cost = np.random.uniform(0, 300)  # 文化引导成本
    new_reputation_gain = np.random.uniform(0, 100)  # 新声誉收益

    # 企业参数
    greenwashing_cost = np.random.uniform(0, 300)  # 漂绿成本
    green_benefit = np.random.normal(100, 30)  # 漂绿收益
    reputation_loss = np.random.normal(50, 10)  # 声誉损失
    passive_greenwashing_benefit = np.random.normal(50, 30)  # 被动漂绿收益

    # 公众参数
    monitoring_cost = np.random.uniform(0, 300)  # 监督成本
    public_loss_benefit = np.random.normal(100, 30)  # 公众效用损失
    public_reputation_loss = np.random.normal(50, 10)  # 欺骗带来的效用损失
    public_reputation_gain = np.random.uniform(0, 100)  # 声誉效用
    public_new_reputation_gain = np.random.uniform(0, 100)  # 新声誉效用
    passive_greenwashing_loss = np.random.normal(100, 30)  # 被动漂绿公众损失
    discovery_prob_single = np.random.uniform(0, 0.5)  # 单方监督发现概率
    discovery_prob_both = np.random.uniform(discovery_prob_single, 1)  # 双方监督发现概率

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

def sample_parameters_4():
    # 政府参数
    government_cost = np.random.uniform(0, 50)  # 监管成本
    fine = np.random.normal(100, 30)  # 罚款力度
    reputation_gain = np.random.uniform(0, 500)  # 声誉收益
    cultural_cost = np.random.uniform(0, 50)  # 文化引导成本
    new_reputation_gain = np.random.uniform(0, 500)  # 新声誉收益

    # 企业参数
    greenwashing_cost = np.random.uniform(0, 50)  # 漂绿成本
    green_benefit = np.random.normal(1000, 30)  # 漂绿收益
    reputation_loss = np.random.normal(50, 10)  # 声誉损失
    passive_greenwashing_benefit = np.random.normal(800, 30)  # 被动漂绿收益

    # 公众参数
    monitoring_cost = np.random.uniform(0, 50)  # 监督成本
    public_loss_benefit = np.random.normal(1000, 30)  # 公众效用损失
    public_reputation_loss = np.random.normal(50, 10)  # 欺骗带来的效用损失
    public_reputation_gain = np.random.uniform(0, 500)  # 声誉效用
    public_new_reputation_gain = np.random.uniform(0, 500)  # 新声誉效用
    passive_greenwashing_loss = np.random.normal(800, 30)  # 被动漂绿公众损失
    discovery_prob_single = np.random.uniform(0, 0.8)  # 单方监督发现概率
    discovery_prob_both = np.random.uniform(discovery_prob_single, 1)  # 双方监督发现概率

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

# 2. 定义收益矩阵函数
def calculate_payoffs(government_strategy, company_strategy, public_strategy, params):
    if government_strategy == "A1":
        if company_strategy == "B1":
            if public_strategy == "C1":
                # 组合1的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] - \
                                    params["government_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - (params["reputation_loss"] + params["fine"]) * params[
                                     "discovery_prob_single"]
                public_payoff = params["public_reputation_gain"] - (1 - params["discovery_prob_single"]) * params[
                    "public_loss_benefit"] - params["discovery_prob_single"] * params["public_reputation_loss"]
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"] - params["government_cost"]
                company_payoff = - params["greenwashing_cost"] - params["fine"] * params["discovery_prob_single"]
                public_payoff = 0
            elif public_strategy == "C3":
                # 组合3的收益计算
                government_payoff = params["discovery_prob_both"] * params["fine"] + params["reputation_gain"] - params[
                    "government_cost"]
                company_payoff = (1 - params["discovery_prob_both"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - params["discovery_prob_both"] * (params["reputation_loss"] + params["fine"])
                public_payoff = params["public_reputation_gain"] - (1 - params["discovery_prob_both"]) * params[
                    "public_loss_benefit"] - params["monitoring_cost"] - params["discovery_prob_both"] * params[
                                    "public_reputation_loss"]
        elif company_strategy == "B2":
            if public_strategy == "C1":
                # 组合1的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] - \
                                    params["government_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["passive_greenwashing_benefit"] - \
                                 params["discovery_prob_single"] * params["fine"]
                public_payoff = params["public_reputation_gain"] - (1 - params["discovery_prob_single"]) * params[
                    "passive_greenwashing_loss"]
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"] - params["government_cost"]
                company_payoff = -params["discovery_prob_single"] * params["fine"]
                public_payoff = 0
            elif public_strategy == "C3":
                # 组合3的收益计算
                government_payoff = params["discovery_prob_both"] * params["fine"] + params["reputation_gain"] - params[
                    "government_cost"]
                company_payoff = (1 - params["discovery_prob_both"]) * params["passive_greenwashing_benefit"] - params[
                    "discovery_prob_both"] * params["fine"]
                public_payoff = params["public_reputation_gain"] - (1 - params["discovery_prob_both"]) * params[
                    "passive_greenwashing_loss"] - params["monitoring_cost"]
        elif company_strategy == "B3":
            if public_strategy == "C1":
                # 组合1的收益计算
                government_payoff = params["reputation_gain"] - params["government_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"]
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = - params["government_cost"]
                company_payoff = 0
                public_payoff = 0
            elif public_strategy == "C3":
                # 组合3的收益计算
                government_payoff = params["reputation_gain"] - params["government_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"] - params["monitoring_cost"]
            # 添加剩余组合的收益计算...
        # 继续定义其他组合情况...
    elif government_strategy == "A2":
        if company_strategy == "B1":
            if public_strategy == "C1":
                # 组合1的收益计算
                government_payoff = 0
                company_payoff = params["green_benefit"] - params["greenwashing_cost"]
                public_payoff = -params["public_loss_benefit"]
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = 0
                company_payoff = - params["greenwashing_cost"]
                public_payoff = 0
            elif public_strategy == "C3":
                # 组合3的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - (params["reputation_loss"] + params["fine"]) * params[
                                     "discovery_prob_single"]
                public_payoff = -(1 - params["discovery_prob_single"]) * params["public_loss_benefit"] - params[
                    "monitoring_cost"] - params["discovery_prob_single"] * params["public_reputation_loss"]
        elif company_strategy == "B2":
            if public_strategy == "C1":
                # 组合1的收益计算
                government_payoff = 0
                company_payoff = params["passive_greenwashing_benefit"]
                public_payoff = -params["passive_greenwashing_loss"]
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = 0
                company_payoff = 0
                public_payoff = 0
            elif public_strategy == "C3":
                # 组合3的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["passive_greenwashing_benefit"] - \
                                 params["discovery_prob_single"] * params["fine"]
                public_payoff = -(1 - params["discovery_prob_single"]) * params["passive_greenwashing_loss"] - params[
                    "monitoring_cost"]
        elif company_strategy == "B3":
            if public_strategy == "C1":
                # 组合1的收益计算
                government_payoff = 0
                company_payoff = 0
                public_payoff = 0
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = 0
                company_payoff = 0
                public_payoff = 0
            elif public_strategy == "C3":
                # 组合3的收益计算
                government_payoff = 0
                company_payoff = 0
                public_payoff = - params["monitoring_cost"]
                # 组合10的收益计算
                # 继续添加收益逻辑...
            # 继续定义其他组合情况...
    elif government_strategy == "A3":
        if company_strategy == "B1":
            if public_strategy == "C1":
                # 组合1的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] + \
                                    params["new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - (params["reputation_loss"] + params["fine"]) * params[
                                     "discovery_prob_single"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_single"]) * params["public_loss_benefit"] - params[
                                    "discovery_prob_single"] * params["public_reputation_loss"]
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] + \
                                    params["new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["green_benefit"] - params[
                    "greenwashing_cost"] - (params["reputation_loss"] + params["fine"]) * params[
                                     "discovery_prob_single"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_single"]) * params["public_loss_benefit"] - params[
                                    "discovery_prob_single"] * params["public_reputation_loss"]
            elif public_strategy == "C3":
                # 组合3的收益计算
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
                # 组合1的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] + \
                                    params["new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["passive_greenwashing_benefit"] - \
                                 params["discovery_prob_single"] * params["fine"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_single"]) * params["passive_greenwashing_loss"]
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = params["discovery_prob_single"] * params["fine"] + params["reputation_gain"] + \
                                    params["new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_single"]) * params["passive_greenwashing_benefit"] - \
                                 params["discovery_prob_single"] * params["fine"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_single"]) * params["passive_greenwashing_loss"]
            elif public_strategy == "C3":
                # 组合3的收益计算
                government_payoff = params["discovery_prob_both"] * params["fine"] + params["reputation_gain"] + params[
                    "new_reputation_gain"] - params["government_cost"] - params["cultural_cost"]
                company_payoff = (1 - params["discovery_prob_both"]) * params["green_benefit"] - params[
                    "discovery_prob_both"] * params["fine"]
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - (
                            1 - params["discovery_prob_both"]) * params["passive_greenwashing_loss"] - params[
                                    "monitoring_cost"]
        elif company_strategy == "B3":
            if public_strategy == "C1":
                # 组合1的收益计算
                government_payoff = params["reputation_gain"] + params["new_reputation_gain"] - params[
                    "government_cost"] - params["cultural_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"]
            elif public_strategy == "C2":
                # 组合2的收益计算
                government_payoff = params["reputation_gain"] + params["new_reputation_gain"] - params[
                    "government_cost"] - params["cultural_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"]
            elif public_strategy == "C3":
                # 组合3的收益计算
                government_payoff = params["reputation_gain"] + params["new_reputation_gain"] - params[
                    "government_cost"] - params["cultural_cost"]
                company_payoff = 0
                public_payoff = params["public_reputation_gain"] + params["public_new_reputation_gain"] - params[
                    "monitoring_cost"]
    return government_payoff, company_payoff, public_payoff


# 初始化每个主体的策略选择概率
initial_probs = {
    "government": {"A1": 1 / 3, "A2": 1 / 3, "A3": 1 / 3},
    "company": {"B1": 1 / 3, "B2": 1 / 3, "B3": 1 / 3},
    "public": {"C1": 1 / 3, "C2": 1 / 3, "C3": 1 / 3}
}


# 动态更新策略选择概率的函数，确保概率非负
def update_strategy_probabilities(current_probs, payoffs, iteration, base_alpha=0.5, temperature=0.5):
    updated_probs = {}
    alpha = base_alpha / (1 + iteration * 0.01)  # 动态学习率，随迭代次数逐渐减小

    # 加入初期扰动，增加初期震荡
    initial_perturbation = np.random.uniform(-0.1, 0.1, len(current_probs))

    for i, (strategy, payoff) in enumerate(payoffs.items()):
        # 使用动态学习率和温度参数，同时引入初期扰动项
        updated_probs[strategy] = max(0, (current_probs[strategy] + alpha * payoff + initial_perturbation[i]) ** (
                    1 / temperature))

    # 归一化，确保总和为 1
    total = sum(updated_probs.values())
    if total > 0:
        for strategy in updated_probs:
            updated_probs[strategy] /= total
    else:
        # 如果总和为零（所有策略概率都为零），则重新均分概率
        num_strategies = len(updated_probs)
        for strategy in updated_probs:
            updated_probs[strategy] = 1 / num_strategies

    return updated_probs


# 3. 蒙特卡洛模拟
def monte_carlo_simulation(num_simulations=500, base_alpha=0.5, temperature=0.5):
    results = []

    government_probs = initial_probs["government"].copy()
    company_probs = initial_probs["company"].copy()
    public_probs = initial_probs["public"].copy()

    for iteration in range(num_simulations):
        params = sample_parameters()

        # 根据当前的策略概率分布选择策略
        government_strategy = np.random.choice(list(government_probs.keys()), p=list(government_probs.values()))
        company_strategy = np.random.choice(list(company_probs.keys()), p=list(company_probs.values()))
        public_strategy = np.random.choice(list(public_probs.keys()), p=list(public_probs.values()))

        # 计算每个主体的收益
        government_payoff, company_payoff, public_payoff = calculate_payoffs(
            government_strategy, company_strategy, public_strategy, params
        )

        total_payoff = government_payoff + company_payoff + public_payoff

        # 更新各主体的策略选择概率，使用改进的 update_strategy_probabilities 函数
        government_probs = update_strategy_probabilities(government_probs, {
            "A1": government_payoff if government_strategy == "A1" else 0,
            "A2": government_payoff if government_strategy == "A2" else 0,
            "A3": government_payoff if government_strategy == "A3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        company_probs = update_strategy_probabilities(company_probs, {
            "B1": company_payoff if company_strategy == "B1" else 0,
            "B2": company_payoff if company_strategy == "B2" else 0,
            "B3": company_payoff if company_strategy == "B3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        public_probs = update_strategy_probabilities(public_probs, {
            "C1": public_payoff if public_strategy == "C1" else 0,
            "C2": public_payoff if public_strategy == "C2" else 0,
            "C3": public_payoff if public_strategy == "C3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        # 记录当前迭代的结果
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

def monte_carlo_simulation_2(num_simulations=500, base_alpha=0.5, temperature=0.5):
    results = []

    government_probs = initial_probs["government"].copy()
    company_probs = initial_probs["company"].copy()
    public_probs = initial_probs["public"].copy()

    for iteration in range(num_simulations):
        params = sample_parameters_2()

        # 根据当前的策略概率分布选择策略
        government_strategy = np.random.choice(list(government_probs.keys()), p=list(government_probs.values()))
        company_strategy = np.random.choice(list(company_probs.keys()), p=list(company_probs.values()))
        public_strategy = np.random.choice(list(public_probs.keys()), p=list(public_probs.values()))

        # 计算每个主体的收益
        government_payoff, company_payoff, public_payoff = calculate_payoffs(
            government_strategy, company_strategy, public_strategy, params
        )

        total_payoff = government_payoff + company_payoff + public_payoff

        # 更新各主体的策略选择概率，使用改进的 update_strategy_probabilities 函数
        government_probs = update_strategy_probabilities(government_probs, {
            "A1": government_payoff if government_strategy == "A1" else 0,
            "A2": government_payoff if government_strategy == "A2" else 0,
            "A3": government_payoff if government_strategy == "A3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        company_probs = update_strategy_probabilities(company_probs, {
            "B1": company_payoff if company_strategy == "B1" else 0,
            "B2": company_payoff if company_strategy == "B2" else 0,
            "B3": company_payoff if company_strategy == "B3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        public_probs = update_strategy_probabilities(public_probs, {
            "C1": public_payoff if public_strategy == "C1" else 0,
            "C2": public_payoff if public_strategy == "C2" else 0,
            "C3": public_payoff if public_strategy == "C3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        # 记录当前迭代的结果
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

def monte_carlo_simulation_3(num_simulations=500, base_alpha=0.5, temperature=0.5):
    results = []

    government_probs = initial_probs["government"].copy()
    company_probs = initial_probs["company"].copy()
    public_probs = initial_probs["public"].copy()

    for iteration in range(num_simulations):
        params = sample_parameters_3()

        # 根据当前的策略概率分布选择策略
        government_strategy = np.random.choice(list(government_probs.keys()), p=list(government_probs.values()))
        company_strategy = np.random.choice(list(company_probs.keys()), p=list(company_probs.values()))
        public_strategy = np.random.choice(list(public_probs.keys()), p=list(public_probs.values()))

        # 计算每个主体的收益
        government_payoff, company_payoff, public_payoff = calculate_payoffs(
            government_strategy, company_strategy, public_strategy, params
        )

        total_payoff = government_payoff + company_payoff + public_payoff

        # 更新各主体的策略选择概率，使用改进的 update_strategy_probabilities 函数
        government_probs = update_strategy_probabilities(government_probs, {
            "A1": government_payoff if government_strategy == "A1" else 0,
            "A2": government_payoff if government_strategy == "A2" else 0,
            "A3": government_payoff if government_strategy == "A3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        company_probs = update_strategy_probabilities(company_probs, {
            "B1": company_payoff if company_strategy == "B1" else 0,
            "B2": company_payoff if company_strategy == "B2" else 0,
            "B3": company_payoff if company_strategy == "B3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        public_probs = update_strategy_probabilities(public_probs, {
            "C1": public_payoff if public_strategy == "C1" else 0,
            "C2": public_payoff if public_strategy == "C2" else 0,
            "C3": public_payoff if public_strategy == "C3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        # 记录当前迭代的结果
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

def monte_carlo_simulation_4(num_simulations=500, base_alpha=0.5, temperature=0.5):
    results = []

    government_probs = initial_probs["government"].copy()
    company_probs = initial_probs["company"].copy()
    public_probs = initial_probs["public"].copy()

    for iteration in range(num_simulations):
        params = sample_parameters_4()

        # 根据当前的策略概率分布选择策略
        government_strategy = np.random.choice(list(government_probs.keys()), p=list(government_probs.values()))
        company_strategy = np.random.choice(list(company_probs.keys()), p=list(company_probs.values()))
        public_strategy = np.random.choice(list(public_probs.keys()), p=list(public_probs.values()))

        # 计算每个主体的收益
        government_payoff, company_payoff, public_payoff = calculate_payoffs(
            government_strategy, company_strategy, public_strategy, params
        )

        total_payoff = government_payoff + company_payoff + public_payoff

        # 更新各主体的策略选择概率，使用改进的 update_strategy_probabilities 函数
        government_probs = update_strategy_probabilities(government_probs, {
            "A1": government_payoff if government_strategy == "A1" else 0,
            "A2": government_payoff if government_strategy == "A2" else 0,
            "A3": government_payoff if government_strategy == "A3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        company_probs = update_strategy_probabilities(company_probs, {
            "B1": company_payoff if company_strategy == "B1" else 0,
            "B2": company_payoff if company_strategy == "B2" else 0,
            "B3": company_payoff if company_strategy == "B3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        public_probs = update_strategy_probabilities(public_probs, {
            "C1": public_payoff if public_strategy == "C1" else 0,
            "C2": public_payoff if public_strategy == "C2" else 0,
            "C3": public_payoff if public_strategy == "C3" else 0
        }, iteration, base_alpha=base_alpha, temperature=temperature)

        # 记录当前迭代的结果
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


# 运行模拟
simulation_results = monte_carlo_simulation(1000)
simulation_results_2 = monte_carlo_simulation_2(1000)
simulation_results_3 = monte_carlo_simulation_3(1000)
simulation_results_4 = monte_carlo_simulation_4(1000)
print(simulation_results.head())

output_path = "output_data/"
os.makedirs(output_path, exist_ok=True)


# 从 simulation_results 提取特征和目标变量
# 特征包括参数，目标变量包括策略选择概率和收益

# 定义特征和目标列
feature_columns = ['government_cost', 'fine', 'reputation_gain', 'cultural_cost', 'new_reputation_gain',
                   'greenwashing_cost', 'green_benefit', 'reputation_loss',
                   'passive_greenwashing_benefit', 'monitoring_cost', 'public_loss_benefit',
                   'public_reputation_loss', 'public_reputation_gain',
                   'public_new_reputation_gain', 'passive_greenwashing_loss',
                   'discovery_prob_single', 'discovery_prob_both']

strategy_columns = ['government_strategy', 'company_strategy', 'public_strategy']
payoff_columns = ['government_payoff', 'company_payoff', 'public_payoff', 'total_payoff']

# 合并多个数据集
all_datasets = pd.concat([simulation_results, simulation_results_2, simulation_results_3, simulation_results_4], ignore_index=True)

# 提取特征和目标变量
X = all_datasets[feature_columns]
y_strategy_prob = all_datasets[strategy_columns]
y_payoff = all_datasets[payoff_columns]

# 类别不平衡处理（以政府策略为例）
# 将数据按类别分开
A1_data = all_datasets[all_datasets['government_strategy'] == 'A1']
A2_data = all_datasets[all_datasets['government_strategy'] == 'A2']
A3_data = all_datasets[all_datasets['government_strategy'] == 'A3']

# 使用过采样，使所有类别数量相等
max_samples = max(len(A1_data), len(A2_data), len(A3_data))
A1_resampled = resample(A1_data, replace=True, n_samples=max_samples, random_state=42)
A2_resampled = resample(A2_data, replace=True, n_samples=max_samples, random_state=42)
A3_resampled = resample(A3_data, replace=True, n_samples=max_samples, random_state=42)

# 合并平衡后的数据集
balanced_data = pd.concat([A1_resampled, A2_resampled, A3_resampled])

# 更新特征和目标数据
X_balanced = balanced_data[feature_columns]
y_strategy_prob_balanced = balanced_data[strategy_columns]
y_payoff_balanced = balanced_data[payoff_columns]

# 网格搜索超参数优化参数
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 定义网格搜索函数
def optimize_model(X, y, param_grid):
    rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

# 政府策略选择模型
best_model_gov, best_params_gov = optimize_model(X_balanced, y_strategy_prob_balanced['government_strategy'], param_grid)
print("Best parameters for Government Strategy Model:", best_params_gov)

X_train, X_test, y_train_gov, y_test_gov = train_test_split(X_balanced, y_strategy_prob_balanced['government_strategy'], test_size=0.2, random_state=42)
best_model_gov.fit(X_train, y_train_gov)
print("Government Strategy Classification Report (Optimized):")
print(classification_report(y_test_gov, best_model_gov.predict(X_test), zero_division=1))

# 企业策略选择模型
best_model_comp, best_params_comp = optimize_model(X_balanced, y_strategy_prob_balanced['company_strategy'], param_grid)
print("Best parameters for Company Strategy Model:", best_params_comp)

X_train, X_test, y_train_comp, y_test_comp = train_test_split(X_balanced, y_strategy_prob_balanced['company_strategy'], test_size=0.2, random_state=42)
best_model_comp.fit(X_train, y_train_comp)
print("Company Strategy Classification Report (Optimized):")
print(classification_report(y_test_comp, best_model_comp.predict(X_test), zero_division=1))

# 公众策略选择模型
best_model_pub, best_params_pub = optimize_model(X_balanced, y_strategy_prob_balanced['public_strategy'], param_grid)
print("Best parameters for Public Strategy Model:", best_params_pub)

X_train, X_test, y_train_pub, y_test_pub = train_test_split(X_balanced, y_strategy_prob_balanced['public_strategy'], test_size=0.2, random_state=42)
best_model_pub.fit(X_train, y_train_pub)
print("Public Strategy Classification Report (Optimized):")
print(classification_report(y_test_pub, best_model_pub.predict(X_test), zero_division=1))


# 设置特征列的简写
feature_short_names = {
    'government_cost': r"$C_g$",
    'fine': 'F',
    'reputation_gain': 'R',
    'cultural_cost': r"$C_{g_2}$",
    'new_reputation_gain': r"$R_2$",
    'greenwashing_cost': r"$C_c$",
    'green_benefit': 'P',
    'reputation_loss': r"$L_c$",
    'passive_greenwashing_benefit': r"$P_2$",
    'monitoring_cost': r"$C_p$",
    'public_loss_benefit': r"$L_p$",
    'public_reputation_loss': r"$L_e$",
    'public_reputation_gain': r"$G_p$",
    'public_new_reputation_gain': r"$G_{p_2}$",
    'passive_greenwashing_loss': r"$L_{p_2}$",
    'discovery_prob_single': r"$\alpha$",
    'discovery_prob_both': r"$\alpha_2$"
}

# 重命名特征列
X_balanced_renamed = X_balanced.rename(columns=feature_short_names)


# 创建SHAP解释器和生成图表的函数
def generate_shap_plots(model, X_data, model_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    # 为每个策略生成 Summary Plot 和 Bar Plot
    for i, strategy_name in enumerate(model.classes_):
        # Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[i], X_data, plot_type="dot", cmap="Spectral", show=False)
        plt.title(f"{model_name} - {strategy_name} Strategy Summary Plot")
        plt.savefig(f"{model_name}_{strategy_name}_summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[i], X_data, plot_type="bar", show=False)
        plt.title(f"{model_name} - {strategy_name} Strategy Bar Plot")
        plt.savefig(f"{model_name}_{strategy_name}_bar_plot.png", dpi=300, bbox_inches='tight')
        plt.close()


# 分别对政府、企业和公众模型生成SHAP图
generate_shap_plots(best_model_gov, X_balanced_renamed, "Government")
generate_shap_plots(best_model_comp, X_balanced_renamed, "Company")
generate_shap_plots(best_model_pub, X_balanced_renamed, "Public")







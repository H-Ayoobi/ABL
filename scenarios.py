"""
Created on Thursday Oct 13 2020
@author: Hamed Ayoobi
َAll rights reserved 2020
"""

import numpy as np


class ScenarioGenerator:
    scenario_type = "first"
    number_of_locations = 6
    number_of_colors = 4
    number_of_concepts = 3
    number_of_recovery_behaviors = 4
    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}
    concepts = {0: 'ball', 1: 'box', 2: 'person'}
    recovery_behaviors = {0: 'push', 1: 'ask', 2: 'alt', 3: 'continue'}
    each_recovery_behavior_occurance = {'push': 0, 'ask': 0, 'alt': 0, 'continue': 0}
    best_recovery_behavior = ""
    rule_set = []
    scenario = []

    def __init__(self, scenario_type):
        self.scenario_type = scenario_type
        self.generate_new_rule_set()
        self.generate_scenario()

    def generate_scenario(self):
        if self.scenario_type == "first":
            self.number_of_locations = 6
            return self.scenario_generator(6)
        elif self.scenario_type == "second":
            self.number_of_locations = 10
            return self.scenario_generator(10)
        elif self.scenario_type == "third":
            self.number_of_locations = 10
            return self.scenario_generator(10)
        # elif self.scenario_type == "fourth":
        #     self.number_of_locations = 4
        #     return self.scenario_generator(4)
        else:
            print("Please choose a correct scenario. You can choose either \'first\' or \'second\'")

    @staticmethod
    def rand_zero_one_vector(n):
        arr = np.zeros(n)
        k = np.random.randint(n - n/4, n)
        arr[:k] = 1
        np.random.shuffle(arr)
        return arr

    def scenario_generator(self, n):
        scenario = []
        locations = self.rand_zero_one_vector(n)
        for loc in locations:
            if loc == 0:
                scenario.append(["Noc", "Noc"])
            else:
                col_con = self.random_combination_of_color_concept()
                scenario.append(col_con)
        self.scenario = scenario
        self.find_best_recovery_behavior()

    def scenario_to_numerical(self):
        numerical = []
        for sen in self.scenario:
            if sen[0] != "Noc":
                numerical.append(list(self.colors.keys())[list(self.colors.values()).index(sen[0])])
                numerical.append(list(self.concepts.keys())[list(self.concepts.values()).index(sen[1])])
            else:
                numerical.append(5)
                numerical.append(5)
        return numerical

    def recovery_behavior_to_numerical(self):
        numerical = list(self.recovery_behaviors.keys())[list(self.recovery_behaviors.values()).index(self.best_recovery_behavior)]
        return numerical

    def recovery_behavior_to_numerical_with_arg(self, recovery_behavior):
        numerical = list(self.recovery_behaviors.keys())[list(self.recovery_behaviors.values()).index(recovery_behavior)]
        return numerical

    def numerical_to_recovery_behavior(self, number):
        recovery_behavior = self.recovery_behaviors[number]
        return recovery_behavior

    def find_best_recovery_behavior(self):
        if self.scenario_type == "first":
            color_concept = f"{self.scenario[0][0]}-{self.scenario[0][1]}"
            self.best_recovery_behavior = self.rule_set[color_concept]
        elif self.scenario_type == "second":
            color_concept = f"{self.scenario[0][0]}-{self.scenario[0][1]},{self.scenario[1][0]}-{self.scenario[1][1]}"
            self.best_recovery_behavior = self.rule_set[color_concept]
        elif self.scenario_type == "third":
            color_concept = f"{self.scenario[0][0]}-{self.scenario[0][1]},{self.scenario[1][0]}-{self.scenario[1][1]},{self.scenario[2][0]}-{self.scenario[2][1]}"
            self.best_recovery_behavior = self.rule_set[color_concept]
        # elif self.scenario_type == "fourth":
        #     color_concept = f"{self.scenario[0][0]}-{self.scenario[0][1]},{self.scenario[1][0]}-{self.scenario[1][1]},{self.scenario[2][0]}-{self.scenario[2][1]},{self.scenario[3][0]}-{self.scenario[3][1]}"
        #     self.best_recovery_behavior = self.rule_set[color_concept]
        self.each_recovery_behavior_occurance[self.best_recovery_behavior] += 1

    def random_combination_of_color_concept(self):
        color_index = np.random.randint(self.number_of_colors)
        concept_index = np.random.randint(self.number_of_concepts)
        return [self.colors[color_index], self.concepts[concept_index]]

    def generate_new_rule_set(self):
        if self.scenario_type == "first":
            rules = {"Noc-Noc": "continue"}
            for _, color in self.colors.items():
                for _, concept in self.concepts.items():
                    recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors -1)
                    rules[f"{color}-{concept}"] = self.recovery_behaviors[recovery_behavior_index]
        elif self.scenario_type == "second":
            rules = {"Noc-Noc,Noc-Noc": "continue"}
            for _, color1 in self.colors.items():
                for _, concept1 in self.concepts.items():
                    for _, color2 in self.colors.items():
                        for _, concept2 in self.concepts.items():
                            recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors -1)
                            rules[f"{color1}-{concept1},{color2}-{concept2}"] = \
                                self.recovery_behaviors[recovery_behavior_index]
            for _, color in self.colors.items():
                for _, concept in self.concepts.items():
                    recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                    rules[f"Noc-Noc,{color}-{concept}"] = self.recovery_behaviors[recovery_behavior_index]
                    recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                    rules[f"{color}-{concept},Noc-Noc"] = self.recovery_behaviors[recovery_behavior_index]
        elif self.scenario_type == "third":
            rules = {"Noc-Noc,Noc-Noc,Noc-Noc": "continue"}
            for _, color1 in self.colors.items():
                for _, concept1 in self.concepts.items():
                    for _, color2 in self.colors.items():
                        for _, concept2 in self.concepts.items():
                            for _, color3 in self.colors.items():
                                for _, concept3 in self.concepts.items():
                                    recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                                    rules[f"{color1}-{concept1},{color2}-{concept2},{color3}-{concept3}"] = \
                                        self.recovery_behaviors[recovery_behavior_index]
            for _, color in self.colors.items():
                for _, concept in self.concepts.items():
                    for _, color2 in self.colors.items():
                        for _, concept2 in self.concepts.items():
                            recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                            rules[f"Noc-Noc,{color}-{concept},{color2}-{concept2}"] = self.recovery_behaviors[
                                recovery_behavior_index]
                            recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                            rules[f"{color}-{concept},Noc-Noc,{color2}-{concept2}"] = self.recovery_behaviors[
                                recovery_behavior_index]
                            recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                            rules[f"{color}-{concept},{color2}-{concept2},Noc-Noc"] = self.recovery_behaviors[
                                recovery_behavior_index]
                            recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                            rules[f"Noc-Noc,{color}-{concept},Noc-Noc"] = self.recovery_behaviors[
                                recovery_behavior_index]
                            recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                            rules[f"Noc-Noc,Noc-Noc,{color2}-{concept2}"] = self.recovery_behaviors[
                                recovery_behavior_index]
                            recovery_behavior_index = np.random.randint(self.number_of_recovery_behaviors - 1)
                            rules[f"{color}-{concept},Noc-Noc,Noc-Noc"] = self.recovery_behaviors[
                                recovery_behavior_index]
        self.rule_set = rules

    def __str__(self):
        str = ""
        str += "------------------------------\n"
        str += f"  loc:\tcolor \t- \tconcept \n"
        str += "------------------------------\n"
        for i in range(self.number_of_locations):
            str += f"\t{i + 1}: \t{self.scenario[i][0]} \t- \t{self.scenario[i][1]}\n"
        str += "-----------------------------------\n"
        str += f" Best recovery behavior: {self.best_recovery_behavior}\n"
        str += "-----------------------------------\n"
        return str


# if __name__ == "__main__":
#     generator = ScenarioGenerator("second")
#     for i in range(10):
#         generator.generate_scenario()
#         print(generator)

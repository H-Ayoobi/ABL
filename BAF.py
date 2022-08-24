"""
Created on Thursday Oct 13 2020
@author: Hamed Ayoobi
َAll rights reserved 2020
"""

import itertools
import numpy as np
import operator



class BAF:
    arguments = []
    attacks = []
    supports = []
    support_weights = {}
    large_subsets = []
    subsets = []
    current_best_recovery_behavior = "Nothing"
    recovery_behaviors = []
    sum_of_weights_for_each_recovery_behavior = {}
    features_weights_for_each_recovery_behavior = []
    combination_feature_weights = {}
    show_rule = False

    def __init__(self, scenario):
        self.arguments = []
        self.attacks = []
        self.supports = []
        self.support_weights = {}
        self.large_subsets = []
        self.current_best_recovery_behavior = "Nothing"
        self.recovery_behaviors = []
        self.sum_of_weights_for_each_recovery_behavior = {}
        self.make_feature_weight_lists(scenario)
        self.recur_subset(range(len(scenario) * 2))
        for itr in self.subsets:
            self.combination_feature_weights[str(itr)] = 0
        self.subsets = []

    def update_baf(self, scenario, best_recovery_behavior):
        self.current_best_recovery_behavior = best_recovery_behavior
        self.add_recovery_behavior()
        self.add_argument(best_recovery_behavior)
        self.add_arguments_from_scenario(scenario)

    def make_feature_weight_lists(self, scenario):
        if self.features_weights_for_each_recovery_behavior == []:
            self.features_weights_for_each_recovery_behavior = np.zeros(len(scenario) * 2)

    def add_recovery_behavior(self):
        if self.current_best_recovery_behavior not in self.recovery_behaviors:
            self.recovery_behaviors.append(self.current_best_recovery_behavior)
            self.add_bidirectional_attack_between_recovery_behaviors()

    def add_bidirectional_attack_between_recovery_behaviors(self):
        for recovery_behavior in self.recovery_behaviors:
            if recovery_behavior != self.current_best_recovery_behavior:
                self.add_attack(self.current_best_recovery_behavior, recovery_behavior)
                self.add_attack(recovery_behavior, self.current_best_recovery_behavior)

    def recur_subset(self, s, l = None):
        if l is None:
            l = len(s)
            self.subsets = []
        if l > 0:
            for x in itertools.combinations(s, l):
                self.subsets.append(list(x))
            self.recur_subset(s, l - 1)

    def add_arguments_from_scenario(self, scenario):
        enumerated_scenarios = self.enumeratea_scenarios(scenario)
        self.recur_subset(enumerated_scenarios)
        for subset in self.subsets:
            self.add_argument(subset)
            self.add_support(subset, self.current_best_recovery_behavior)

    def enumeratea_scenarios(self,scenario):
        enumerated_scenarios = []
        for idx in range(len(scenario)):
            if scenario[idx][0] != 'Noc':
                enumerated_scenarios.append([idx, scenario[idx][0]])
                enumerated_scenarios.append([idx, scenario[idx][1]])
            else:
                enumerated_scenarios.append([idx, scenario[idx][0], scenario[idx][1]])
        return enumerated_scenarios

    def extract_arguments_from_scenario(self, scenario):
        enumerated_scenarios = self.enumeratea_scenarios(scenario)
        self.recur_subset(enumerated_scenarios)

    def compute_sum_of_weights_for_each_recovery_behavior(self, show_rule):
        sum_of_weights = {}
        max_support_for_recovery_behavior = {}
        self.sum_of_weights_for_each_recovery_behavior = sum_of_weights
        for subset in self.subsets:
            sbt = self.arg_to_combination_numbers(subset)
            for recovery_behavior in self.recovery_behaviors:
                support_relation = f"{subset}->{recovery_behavior}"
                if (recovery_behavior in sum_of_weights) and (support_relation in self.support_weights):
                    # sum_of_weights[recovery_behavior] += self.support_weights[support_relation] * (self.combination_feature_weights[str(sbt)])
                    if (self.support_weights[support_relation] * self.combination_feature_weights[str(sbt)]) > sum_of_weights[recovery_behavior]:
                        max_support_for_recovery_behavior[recovery_behavior] = support_relation
                    sum_of_weights[recovery_behavior] = max(self.support_weights[support_relation] * (self.combination_feature_weights[str(sbt)]),
                                                            sum_of_weights[recovery_behavior])
                elif support_relation in self.support_weights:
                    sum_of_weights[recovery_behavior] = self.support_weights[support_relation] * (self.combination_feature_weights[str(sbt)])
                    max_support_for_recovery_behavior[recovery_behavior] = support_relation
                elif recovery_behavior in sum_of_weights:
                    pass
                else:
                    sum_of_weights[recovery_behavior] = -1000

        self.sum_of_weights_for_each_recovery_behavior = sum_of_weights

        if show_rule:
            #show which rule has been used for finding the best recovery behavior at each step
            if self.sum_of_weights_for_each_recovery_behavior != {} and max_support_for_recovery_behavior != {}:
                max_key = max(self.sum_of_weights_for_each_recovery_behavior.items(), key=operator.itemgetter(1))[0]
                print(max_support_for_recovery_behavior[max_key])


    def find_recovery_behavior_with_highest_sum_of_support(self):
        max = -10000
        recovery_behavior_with_highest_sum = ""
        for recovery_behavior, sum_weight in self.sum_of_weights_for_each_recovery_behavior.items():
            if sum_weight > max:
                max = sum_weight
                recovery_behavior_with_highest_sum = recovery_behavior

        return recovery_behavior_with_highest_sum


    def add_argument(self, argument):
        if argument not in self.arguments:
            self.arguments.append(argument)

    def add_attack(self, argument1, argument2):
        if [argument1, argument2] not in self.attacks:
            self.attacks.append([argument1, argument2])

    def add_support(self, argument1, argument2):
        if [argument1, argument2] not in self.supports:
            self.supports.append([argument1, argument2])
        self.update_support_weight(argument1, argument2)


    def arg_to_combination_numbers(self, argument1):
        lst = []
        for arg in argument1:
            if len(arg)==3:
                lst.append(arg[0]*2)
                lst.append(arg[0]*2 + 1)
            elif len(arg) ==2:
                if arg[1] in ['red', 'green', 'blue', 'yellow']:
                    lst.append(arg[0]*2)
                else:
                    lst.append(arg[0]*2 + 1)
        return lst


    def update_support_weight(self, argument1, argument2,):
        args_set = self.arg_to_combination_numbers(argument1)
        if (f"{argument1}->{argument2}" in self.support_weights):
            self.combination_feature_weights[str(args_set)] += 1

        repetitions = {}
        sum = 1
        for recovery_behavior in self.recovery_behaviors:
            if recovery_behavior != argument2:
                if f"{argument1}->{recovery_behavior}" in self.support_weights:
                    repetitions[recovery_behavior] = 1
                    sum += 1
                    self.combination_feature_weights[str(args_set)] -= 1

        self.support_weights[f"{argument1}->{argument2}"] = 1 / float(sum) #if sum == 1 else 0
        for recovery_behavior, repetition in repetitions.items():
            if repetition == 1:
                self.support_weights[f"{argument1}->{recovery_behavior}"] = 1 / float(sum) #if sum == 1 else 0

    def generate_second_guess(self, scenario, show_rule=False):
        self.extract_arguments_from_scenario(scenario)
        self.compute_sum_of_weights_for_each_recovery_behavior(show_rule)
        return self.find_recovery_behavior_with_highest_sum_of_support()


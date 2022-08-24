"""
Created on Thursday Oct 13 2020
@author: Hamed Ayoobi
َAll rights reserved 2020
"""


import scenarios
import BAF
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

if __name__ == "__main__":
    number_of_attempts = 200
    number_of_iterations = 3
    all_TPs = np.zeros((number_of_iterations,number_of_attempts))
    all_DT_TPs = np.zeros((number_of_iterations, number_of_attempts))
    scenario_type = "first" #options are "first" and "second"
    times = []
    for itr in range(number_of_iterations):
        gen = scenarios.ScenarioGenerator(scenario_type)
        baf = BAF.BAF(gen.scenario)
        saved_scenarios_in_memory_for_other_approaches = []
        saved_best_recovery_behaviors_in_memory_for_other_approaches = []
        TP = 0
        DT_TP = 0
        start = timer()
        for attempt in range(number_of_attempts):
            # start = timer()
            gen.generate_scenario()
            # print(gen)
            guess = baf.generate_second_guess(gen.scenario, show_rule=False)
            baf.update_baf(gen.scenario, gen.best_recovery_behavior)
            if gen.best_recovery_behavior == guess:
                TP += 1
            all_TPs[itr, attempt] = TP
            #Decision tree with sklearn
            if saved_scenarios_in_memory_for_other_approaches:
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(saved_scenarios_in_memory_for_other_approaches, saved_best_recovery_behaviors_in_memory_for_other_approaches)
                dt_guess = gen.numerical_to_recovery_behavior(clf.predict([gen.scenario_to_numerical()])[0])
                if dt_guess == gen.best_recovery_behavior:
                    DT_TP += 1
                all_DT_TPs[itr, attempt] = DT_TP
                # tree.plot_tree(clf)
                # plt.show()

            saved_scenarios_in_memory_for_other_approaches.append(gen.scenario_to_numerical())
            saved_best_recovery_behaviors_in_memory_for_other_approaches.append(gen.recovery_behavior_to_numerical())
            # print(f"time: {timer()-start} s")
            print(f"{attempt}:Our: {TP}, Decition tree: {DT_TP}")

        one_iteration_time = timer() - start
        times.append(one_iteration_time)
        print("average_per_iteration_time: ", np.mean(times))

        print(f"Our model true positives: {TP}")
        print(f"Decision Tree's true positives: {DT_TP}")
        print(f"Overal: {np.mean(all_TPs[:itr+1],axis=0)}")
        print(f"Overal accuracy: {np.mean(all_TPs[:itr+1],axis=0)/(np.array(range(number_of_attempts)) + 1)}")
        print(f"Overal DT: {np.mean(all_DT_TPs,axis=0)}")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        print(f"Overal DT accuracy: {np.mean(all_DT_TPs[:itr+1],axis=0)/(np.array(range(number_of_attempts)) + 1)}")
        ax.plot(range(number_of_attempts), np.mean(all_TPs[:itr+1], axis=0)/(np.array(range(number_of_attempts)) + 1), 'g-', label="ABL (our)")
        ax.plot(range(number_of_attempts), np.mean(all_DT_TPs[:itr+1], axis=0)/(np.array(range(number_of_attempts)) + 1), 'r-', label="Decision Tree")
        ax.legend()
        fig.savefig(f"plots/fig_{itr}.png", bbox_inches='tight')
        if itr != number_of_iterations - 1: #only the final plot will be shown (all the plots has been saved)
            plt.close(fig)
    plt.show()




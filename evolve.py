from utils import StatusUpdateTool, Utils, Log
from genetic.population import Population
from genetic.evaluate import FitnessEvaluate
from genetic.crossover_and_mutation import CrossoverAndMutation
from genetic.selection_operator import Selection
from genetic.nsga_two import NSGAII
import numpy as np
import copy

class EvolveSNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None

    def initialize_population(self):
        StatusUpdateTool.begin_evolution() # set evolution_status to 1
        pops = Population(params, 0)
        pops.initialize()
        self.pops = pops


        Utils.save_population_at_begin(str(pops), 0) # write initialize population info to the begin_00.txt

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()

    def crossover_and_mutation(self):
        cm = CrossoverAndMutation(self.params['genetic_prob'][0], self.params['genetic_prob'][1], Log, self.pops.individuals, self.pops, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process()
        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        combine_individuals = self.parent_pops.individuals + self.pops.individuals
        # print(combine_individuals)
        combine_pops = Population(self.pops.params, -1)
        combine_pops.pop_size = combine_pops.pop_size * 2
        combine_pops.create_from_offspring(combine_individuals)
        nsga = NSGAII()
        combine_pops.front = nsga.fast_nodominate_sort(combine_individuals)
        for Fi in combine_pops.front:
            nsga.crowding_dist(Fi, combine_individuals)

        count_ = 0
        offspring_individuals = []
        offspring_size = self.params['pop_size']
        for each_front in combine_pops.front:
            if count_ >= offspring_size:
                break
            remain_empty_pop_length = offspring_size - count_
            current_front_length = len(each_front)
            if remain_empty_pop_length < current_front_length:  # remain empty pop length can not load current front individuals, neet to sort according to crowding distance.
                # sort according to crowding distance
                distance_list = []
                for id in each_front:
                    distance = [id, float(combine_individuals[id].crowd_distance)]
                    distance_list.append(distance)
                print('original distance list:', distance_list)
                distance_list.sort(key=lambda x: x[1], reverse=True)
                print('sorted distance list:', distance_list)
                each_front.clear()
                for distance in distance_list:
                    each_front.append(distance[0])
                print('sorted front list:', each_front)
            for id in each_front:
                if count_ < offspring_size:
                    offspring_individuals.append(combine_individuals[id])
                    count_ += 1
                else:
                    break

        next_gen_pops = Population(self.pops.params, self.pops.gen_no + 1)
        for indi in offspring_individuals:
            indi.reset_Sp_Np()
        next_gen_pops.create_from_offspring(offspring_individuals)
        self.pops = next_gen_pops
        self.pops.front = []

        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def do_work(self, max_gen):
        Log.info('*'*25)
        # the step 1
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation'%(gen_no))
                pops = Utils.load_population('begin', gen_no)
                self.pops = pops
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            # first use
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(gen_no))
        gen_no += 1
        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            #step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation'%(curr_gen))
            self.crossover_and_mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation'%(curr_gen))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(curr_gen))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(curr_gen))

            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection'%(curr_gen))

        StatusUpdateTool.end_evolution() # set evolution to 0

if __name__ == '__main__':
    params = StatusUpdateTool.get_init_params()
    evoCNN = EvolveSNN(params)
    evoCNN.do_work(max_gen=20)


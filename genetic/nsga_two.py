class NSGAII(object):
    def __init__(self):
        pass

    def fast_nodominate_sort(self, population):
        F = []
        i = 1
        F1 = self.cpt_F1_dominate(population)
        while len(F1) != 0:
            F.append(F1)
            Q = []
            for pi in F1:
                p = population[pi] # a single individual
                for q in p.Sp: # sp is a dominate indis list
                    # print(q, p.Sp)
                    one_q = population[q]
                    one_q.Np = one_q.Np - 1
                    if one_q.Np == 0:
                        one_q.p_rank = i + 1
                        Q.append(q)
            i = i + 1
            F1 = Q
        return F

    # P is the list of population
    def cpt_F1_dominate(self, population):
        F1 = []
        for j, p in enumerate(population):
            for i, q in enumerate(population):
                if j != i:
                    if self.is_dominate(p, q):
                        if i not in p.Sp:
                            p.Sp.append(i)
                    elif self.is_dominate(q, p):
                        p.Np = p.Np + 1
            if p.Np == 0:
                p.p_rank = 1
                F1.append(j)
        return F1

    # F_value is the list of function1 and function2
    def is_dominate(self, a, b):
        a_f = a.get_F_value()
        b_f = b.get_F_value()
        i = 0
        for av, bv in zip(a_f, b_f):
            if av < bv:
                i = i + 1
            if av > bv:
                return False
        if i != 0:
            return True
        return False

    def crowding_dist(self, Fi, population):
        f_max = population[Fi[0]].get_F_value()
        f_min = population[Fi[0]].get_F_value()
        f_num = len(f_max)
        for p in Fi:
            population[p].crowd_distance = 0
            for fm in range(f_num):
                if population[p].get_F_value()[fm] > f_max[fm]:
                    f_max[fm] = population[p].get_F_value()[fm]
                if population[p].get_F_value()[fm] < f_min[fm]:
                    f_min[fm] = population[p].get_F_value()[fm]
        Fi_len = len(Fi)
        for m in range(f_num):
            Fi = self.sort_func(Fi, m, population)
            population[Fi[0]].crowd_distance = 1000000
            population[Fi[Fi_len - 1]].crowd_distance = 1000000 # boundary points is always selected.
            for f in range(1, Fi_len - 1):
                a = population[Fi[f + 1]].get_F_value()[m] - population[Fi[f - 1]].get_F_value()[m]
                b = f_max[m] - f_min[m]
                if b == 0:
                    population[Fi[f]].crowd_distance = -1
                else:
                    population[Fi[f]].crowd_distance = population[Fi[f]].crowd_distance + a / b

    def sort_func(self, Fi, m, population):
        FL = len(Fi)
        for i in range(FL - 1):
            p = Fi[i]
            for j in range(i + 1, FL):
                q = Fi[j]
                if p != q and population[p].get_F_value()[m] > population[q].get_F_value()[m]:
                    Fi[i], Fi[j] = Fi[j], Fi[i]
        return Fi

    def get_individual(self, num, population):
        return population[num]

# def test_population():
#     params = {}
#     params['pop_size'] = 20
#     params['min_conv'] = 10
#     params['max_conv'] = 15
#     params['min_pool'] = 0
#     params['max_pool'] = 0
#     params['max_len'] = 20
#     # params['conv_kernel'] = [1, 2, 3]
#     # params['conv_stride'] = [1,2, 3]
#     # params['pool_kernel'] = 2
#     # params['pool_stride'] = 2
#     params['image_channel'] = 3
#     params['output_channel'] = [64, 128, 256, 512]
#     params['min_lif'] = 3
#     params['max_lif'] = 15
#     pop = population.Population(params, 0)
#     pop.initialize()
#     # print(pop)
#     return pop
#
# if __name__ == '__main__':
#     population = test_population()
#     nsga = NSGAII()
#     F = nsga.fast_nodominate_sort(population=population.individuals)
#     population.front = F
#     print(population.front)
#     Fi = F[0]
#     for Fi in F:
#         nsga.crowding_dist(Fi, population.individuals)
#     for indi in population.individuals:
#         print(indi.crowd_distance)

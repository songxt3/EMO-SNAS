import random

import numpy as np
import hashlib
import copy

"""
node type 1 is LIFNode layer
node type 2 is pooling layer
"""

class Unit(object):
    def __init__(self, number):
        self.number = number

# TODO:maybe could use IF unit for choosing
class LIFUnit(Unit):
    def __init__(self, number, in_channel, out_channel, fire, backward):
        super().__init__(number)
        self.type = 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.backward = backward
        self.fire = fire
        self.neuron = 0 # default value 0 is LIF node


# TODO:not used current, 2022/5/16
class PoolUnit(Unit):
    def __init__(self, number):
        super().__init__(number)
        self.type = 2


class Individual(object):
    def __init__(self, params, indi_no):
        self.inv_acc = -1.0
        self.spike_num = -1.0
        self.crowd_distance = -1.0
        self.id = indi_no # for record the id of current individual
        self.number_id = 0 # for record the latest number of basic unit
        self.min_conv = params['min_conv']
        self.max_conv = params['max_conv']
        self.min_lif = params['min_lif']
        self.max_lif = params['max_lif']
        self.min_pool = params['min_pool']
        self.max_pool = params['max_pool']
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_channles = params['output_channel']
        self.units = []
        self.Sp = []
        self.Np = 0

    def get_F_value(self):
        return [float(self.inv_acc), float(self.spike_num)]

    def reset_performance(self):
        self.inv_acc = -1.0
        self.spike_num = -1.0
        self.crowd_distance = -1.0

    def reset_Sp_Np(self):
        self.Sp = []
        self.Np = 0

    def initialize(self):
        num_lifnode = np.random.randint(self.min_lif, self.max_lif+1)
        num_pool = np.random.randint(self.min_pool, self.max_pool+1)
        availabel_positions = list(range(num_lifnode))
        np.random.shuffle(availabel_positions)
        select_positions = np.sort(availabel_positions[0:num_pool])
        all_positions = []
        for i in range(num_lifnode):
            all_positions.append(1)
            for j in select_positions:
                if j == i:
                    all_positions.append(2)
                    break
        # initialize the layers based on their positions
        input_channel = self.image_channel
        for i in all_positions:
            if i == 1:
                lifnode = self.init_a_lifnode(_number=None, _in_channel=input_channel, _out_channel=None, _fire=True, _backward=False)
                input_channel = lifnode.out_channel
                self.units.append(lifnode)
            elif i == 2:
                pool = self.init_a_pool(_number=None)
                self.units.append(pool)

    def init_a_lifnode(self, _number, _in_channel, _out_channel, _fire, _backward):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]

        lifnode = LIFUnit(number, _in_channel, out_channel, _fire, _backward)
        return lifnode

    def init_a_pool(self, _number):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        pool = PoolUnit(number)
        return pool

    def uuid(self):
        _str = []
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('lifnode')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('fire:%d' % (unit.fire))

            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d' % (unit.number))
            _str.append('%s%s%s'%('[', ','.join(_sub_str), ']'))
        _final_str_ = '-'.join(_str)
        _final_utf8_str_= _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_

    def __str__(self):
        _str = []
        _str.append('indi:%s'%(self.id))
        _str.append('inv_acc:%.5f'%(self.inv_acc))
        _str.append('spike_num:%.5f'%(self.spike_num))
        _str.append('crowd_distance:%.5f'%(self.crowd_distance))
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('lifnode')
                _sub_str.append('number:%d'%(unit.number))
                _sub_str.append('in:%d'%(unit.in_channel))
                _sub_str.append('out:%d'%(unit.out_channel))
                _sub_str.append('fire:%d'%(unit.fire))
                _sub_str.append('backward:%d'%(unit.backward))

            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d' % (unit.number))
            _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
        return '\n'.join(_str)


class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0 # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []
        self.front = []
        self.distance = []

    def initialize(self):
        # create pop_size number of individuals
        for _ in range(self.pop_size):
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-'*100)
        return '\n'.join(_str)


def test_individual():
    params = {}
    params['min_conv'] = 30
    params['max_conv'] = 40
    params['min_pool'] = 3
    params['max_pool'] = 4
    params['max_len'] = 20
    params['image_channel'] = 3
    params['output_channel'] = [64, 128, 256, 512]
    params['min_lif'] = 3
    params['max_lif'] = 15
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())

def test_population():
    params = {}
    params['pop_size'] = 20
    params['min_conv'] = 10
    params['max_conv'] = 15
    params['min_pool'] = 0
    params['max_pool'] = 0
    params['max_len'] = 20
    params['image_channel'] = 3
    params['output_channel'] = [64, 128, 256, 512]
    params['min_lif'] = 3
    params['max_lif'] = 15
    pop = Population(params, 0)
    pop.initialize()
    print(pop)
    return pop



if __name__ == '__main__':

    # test_individual()
    test_population()
    b = []
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    for i in range(0, len(a), 2):
        b.append(a[i:i + 2])
    print(b)





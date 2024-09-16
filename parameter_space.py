class ParameterSpace:

    def __init__(self, target_props):

        self.target_props = target_props
        # Number of dimensions in the parameter space (e.g. number of galaxy properties)
        self.Ndims = len(target_props)

        # Possible combinations (pairs) of dimensions (2D subspaces)
        # indices
        # [:4] hard coded for astro diagrams
        self.pairs = [[i, j] for i in range(self.Ndims) for j in range(i + 1, self.Ndims)][:4]
        pair_name_list = []
        # names (str)
        for pair_ind in self.pairs:
            pair_name_list.append('{}_{}'.format(self.target_props[pair_ind[0]],
                                                 self.target_props[pair_ind[1]]))
        self.pair_name_list = pair_name_list

        # Name of event space: combine the name of all target properties (dimensions)
        if len(self.target_props) == 1:
            prop = target_props[0]
        else:
            prop = self.target_props[0] + '_'
            for i in self.target_props[1:]:
                prop = prop + i
                if i != self.target_props[-1]:
                    prop = prop + '_'
        self.name_of_event_space = prop

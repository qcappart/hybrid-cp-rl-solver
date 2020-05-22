from pdtsp import abs_norm


class State:
    def __init__(self, instance, must_visit, last_visited, cur_load, tour):
        """
        Build a State.
        Note that the set of valid actions correspond to the must_visit part of the state
        :param instance: the problem instance considered
        :param must_visit: cities that still have to be visited.
        :param last_visited: the current location
        :param cur_load: the current load (vector of commodities weights)
        :param tour: the tour that is currently done
        """

        self.instance = instance
        self.must_visit = must_visit
        self.last_visited = last_visited
        self.cur_load = cur_load
        self.tour = tour

    def step(self, action):
        """
        Performs the transition function of the DP model
        :param action: the action selected
        :return: the new state wrt the transition function on the current state T(s,a) = s'
        """

        new_must_visit = self.must_visit - set([action])
        new_last_visited = action
        new_cur_load = self.cur_load + self.instance.pickup_constraints[action]

        assert all(new_cur_load >= 0) and abs_norm(new_cur_load) <= self.instance.capacity, "Tried to step with an invalid action !"
        
        new_tour = self.tour + [new_last_visited]

        #  Application of the validity conditions and the pruning rules before creating the new state
        new_must_visit = self.prune_invalid_actions(new_must_visit, new_last_visited, new_cur_load)
        new_must_visit = self.prune_dominated_actions(new_must_visit, new_cur_load)

        new_state = State(self.instance, new_must_visit, new_last_visited, new_cur_load, new_tour)

        return new_state

    def is_done(self):
        """
        :return: True iff there is no remaining actions
        """

        return len(self.must_visit) == 0

    def is_success(self):
        """
        :return: True iff there is the tour is fully completed
        """

        return len(self.tour) == self.instance.n_city

    def prune_invalid_actions(self, new_must_visit, new_last_visited, new_cur_load):
        """
        Validity condition: Keep only the cities that can fit in the time windows according to the travel time.
        :param new_must_visit: the cities that we still have to visit
        :param new_last_visited: the city where we are
        :param new_cur_time: the current time
        :return:
        """

        pruned_must_visit = [a for a in new_must_visit if
                             all(new_cur_load + self.instance.pickup_constraints[a] >= 0) and abs_norm(new_cur_load + self.instance.pickup_constraints[a]) <= self.instance.capacity]

        return set(pruned_must_visit)

    def prune_dominated_actions(self, new_must_visit, new_cur_load):
        """
        Pruning dominated actions: We remove all the cities which have an uncarried commodity to be delivered at
        :param new_must_visit: the cities that we still have to visit
        :param new_cur_load: the current load
        :return:
        """

        pruned_must_visit = [a for a in new_must_visit if all(new_cur_load + self.instance.pickup_constraints[a] >= 0)]

        return set(pruned_must_visit)

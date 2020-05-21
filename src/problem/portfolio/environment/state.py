
class State:

    def __init__(self, instance, weight, stage, available_action, items_taken):
        """
         Build a State
         :param instance: problem instance considered
         :param weight: cumulated weight of the items inside the portfolio
         :param stage: current stage
         :param available_action: actions possible (insert or not) for the current items
         :param items_taken: keep track of the previous decisions
        """

        self.instance = instance
        self.weight = weight
        self.stage = stage
        self.available_action = available_action
        self.items_taken = items_taken

    def step(self, action):
        """
        Performs the transition function of the DP model
        :param action: the action selected
        :return: the new state wrt the transition function on the current state T(s,a) = s'
        """

        collected_profit = self.instance.moment_factors[0] * action * self.instance.means[self.stage]
        new_weight = self.weight + action * self.instance.weights[self.stage]

        new_items_taken = self.items_taken + [action]

        # No action for the last stage
        if self.stage + 1 == self.instance.n_item:
            new_available_action = set()

        # Not possible to insert the item if its insertion exceed the portfolio capacity
        elif new_weight + self.instance.weights[self.stage + 1] > self.instance.capacity:
            new_available_action = set([0])
        else:
            new_available_action = set([0, 1])

        new_stage = self.stage + 1

        new_state = State(self.instance, new_weight, new_stage, new_available_action, new_items_taken)

        assert new_weight <= self.instance.capacity

        return new_state, collected_profit

    def is_done(self):
        """
        :return: True iff there is no remaining actions
        """

        return self.stage == self.instance.n_item
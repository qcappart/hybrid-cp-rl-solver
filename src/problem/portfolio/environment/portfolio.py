import random

class Portfolio:

    def __init__(self, n_item, capacity, weights, means, deviations, skewnesses, kurtosis, moment_factors):
        """
        Create an instance of the 4-moment portfolio optimization problem
        :param n_item: number of items to add (or not) in the portfolio
        :param capacity: maximal capacity of the portfolio
        :param weights: list of weights of the items
        :param means: list of mean values of the items
        :param deviations: list of deviation values of the items
        :param skewnesses: list of skewness vales of the items
        :param kurtosis: list of kurtosis values of the items
        :param moment_factors: list of factor coefficient for the moments (4 values)
        """

        self.n_item = n_item
        self.capacity = capacity
        self.weights = weights
        self.means = means
        self.deviations = deviations
        self.skewnesses = skewnesses
        self.kurtosis = kurtosis
        self.moment_factors = moment_factors

        self.index_list = list(range(n_item))

        self.object_list = list(zip(self.index_list, self.weights, self.means, self.deviations, self.skewnesses, self.kurtosis))


    @staticmethod
    def generate_random_instance(n_item, lb, ub, capacity_ratio, moment_factors, is_integer_instance, seed):
        """
        :param n_item: number of items to add (or not) in the portfolio
        :param lb: lowest value allowed for generating the moment values
        :param ub: highest value allowed for generating the moment values
        :param capacity_ratio: capacity of the instance is capacity_ratio * (sum of all the item weights)
        :param moment_factors: list of factor coefficient for the moments (4 values)
        :param is_integer_instance: True if we want the powed moment values to be integer
        :param seed: seed used for generating the instance. -1 means no seed (instance is random)
        :return: a 4-moment portfolio optimization problem randomly generated using the parameters
        """

        rand = random.Random()

        if seed != -1:
            rand.seed(seed)

        weights = [rand.uniform(lb, ub) for _ in range(n_item)]
        means = [rand.uniform(lb, ub) for _ in range(n_item)]
        deviations = [rand.uniform(lb, means[i]) ** 2 for i in range(n_item)]
        skewnesses = [rand.uniform(lb, means[i]) ** 3 for i in range(n_item)]
        kurtosis = [rand.uniform(lb, means[i]) ** 4 for i in range(n_item)]

        if is_integer_instance:
            weights = [int(x) for x in weights]
            means = [int(x) for x in means]
            deviations = [int(x) for x in deviations]
            skewnesses = [int(x) for x in skewnesses]
            kurtosis = [int(x) for x in kurtosis]

        capacity = int(capacity_ratio * sum(weights))
        final_moment_factors = []

        for i in range(4):

            if moment_factors[i] == -1:
                moment_coeff = rand.randint(1, 15)
                final_moment_factors.append(moment_coeff)
            else:
                final_moment_factors.append(moment_factors[i])

        return Portfolio(n_item, capacity, weights, means, deviations, skewnesses, kurtosis, moment_factors)


    @staticmethod
    def generate_dataset(size, n_item, lb, ub, capacity_ratio, moment_factors, is_integer_instance, seed):
        """
        Generate a dataset of instance
        :param size: the size of the data set
        :param n_item: number of items to add (or not) in the portfolio
        :param lb: lowest value allowed for generating the moment values
        :param ub: highest value allowed for generating the moment values
        :param capacity_ratio: capacity of the instance is capacity_ratio * (sum of all the item weights)
        :param moment_factors: list of factor coefficient for the moments (4 values)
        :param is_integer_instance: True if we want the powed moment values to be integer
        :param seed: seed used for generating the instance. -1 means no seed (instance is random)
        :return: a dataset of '#size' 4-moment portfolio instances randomly generated using the parameters
        """

        dataset = []
        for i in range(size):
            new_instance = Portfolio.generate_random_instance(n_item=n_item, lb=lb, ub=ub,
                                                              capacity_ratio=capacity_ratio,
                                                              moment_factors=moment_factors,
                                                              is_integer_instance=is_integer_instance,
                                                              seed=seed)
            dataset.append(new_instance)
            seed += 1

        return dataset
#include <chrono>
#include <random>
#include <math.h>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/set.hh>
#include <gecode/search.hh>
#include <gecode/minimodel.hh>
#include <gecode/float.hh>

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <map>
#include <limits>
#include<string>
#include <cxxopts.hpp>

using namespace Gecode;
using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

std::chrono::time_point<std::chrono::system_clock> start;

std::map<std::tuple<int, int>, std::vector<double>> predict_cache;



Rnd r(1U);
auto t = time(0);
std::default_random_engine generator(0);

FloatVal best_cost = -100000;

class OptionsPortfolio: public Options {
public:
  int n_item; /// Parameters to be given on the command line
  bool cache;
  float capacity_ratio;
  int lambda_1;
  int lambda_2;
  int lambda_3;
  int lambda_4;
  float temperature;
  bool discrete_coeffs;

  OptionsPortfolio(const char* s, int n_item0, bool cache0, float capacity_ratio0,
                   int lambda_10, int lambda_20, int lambda_30, int lambda_40, float temperature0, bool discrete_coeffs0)
    : Options(s), n_item(n_item0), cache(cache0), capacity_ratio(capacity_ratio0),
      lambda_1(lambda_10), lambda_2(lambda_20), lambda_3(lambda_30), lambda_4(lambda_40), temperature(temperature0), discrete_coeffs(discrete_coeffs0) {}
};

class Portfolio_DP : public FloatMaximizeScript {
private:


    std::vector<int> weight_list;
    std::vector<int> mean_list;
    std::vector<int> deviation_list;
    std::vector<int> skewness_list;
    std::vector<int> kurtosis_list;


    int capacity;
    int lambda_1;
    int lambda_2;
    int lambda_3;
    int lambda_4;

    IntVarArray acc_weight;
    IntVarArray take_item;

    FloatVar total_profit;

    int n_item;
    float temperature;
    py::object to_run;
    bool use_cache;
    int seed;

public:
    enum ModelType {RL_BAB, RL_DQN, RL_PPO};
    /* Constructor */
    Portfolio_DP(const OptionsPortfolio& opt) : FloatMaximizeScript(opt) {
        this->n_item = opt.n_item;
        this->lambda_1 = opt.lambda_1;
        this->lambda_2 = opt.lambda_2;
        this->lambda_3 = opt.lambda_3;
        this->lambda_4 = opt.lambda_4;
        this->seed = opt.seed();
        this->temperature = opt.temperature;

        this->use_cache = opt.cache;


        start = std::chrono::system_clock::now();

        string mode = "";

        if (opt.model() == RL_DQN || opt.model() == RL_BAB) {
            mode = "dqn";
        }
        else if(opt.model() == RL_PPO) {
            mode = "ppo";
        }

        std::stringstream stream;
        stream << std::fixed << std::setprecision(1) << opt.capacity_ratio;

        string model_folder = "./selected-models/" + mode + "/portfolio/n-item-" + std::to_string(this->n_item) +
                              "/capacity-ratio-" + stream.str() +
                              "/moment-factors-" + std::to_string(opt.lambda_1) + "-" + std::to_string(opt.lambda_2) + "-" +
                              std::to_string(opt.lambda_3)+ "-" + std::to_string(opt.lambda_4);

        auto to_run_module = py::module::import("src.problem.portfolio.solving.solver_binding");
        to_run = to_run_module.attr("SolverBinding")(model_folder, this->n_item, opt.capacity_ratio, opt.lambda_1, opt.lambda_2,
                                                      opt.lambda_3, opt.lambda_4, opt.discrete_coeffs, this->seed, mode);

        auto weight_list_python = to_run.attr("get_weight_list")();
        weight_list = weight_list_python.cast<std::vector<int>>();

        auto mean_list_python = to_run.attr("get_mean_list")();
        mean_list = mean_list_python.cast<std::vector<int>>();

        auto deviation_list_python = to_run.attr("get_deviation_list")();
        deviation_list = deviation_list_python.cast<std::vector<int>>();

        auto skewness_list_python = to_run.attr("get_skewness_list")();
        skewness_list = skewness_list_python.cast<std::vector<int>>();

        auto kurtosis_list_python = to_run.attr("get_kurtosis_list")();
        kurtosis_list = kurtosis_list_python.cast<std::vector<int>>();


        capacity = to_run.attr("get_capacity")().cast<int>();

        /* Parameters */
        IntArgs weight_list_args(n_item);
        FloatValArgs mean_list_args(n_item);
        FloatValArgs deviation_list_args(n_item);
        FloatValArgs skewness_list_args(n_item);
        FloatValArgs kurtosis_list_args(n_item);


        for (int i = 0; i <  n_item; i++) {
            weight_list_args[i] = weight_list[i];
            mean_list_args[i] = mean_list[i];
            deviation_list_args[i] = deviation_list[i];
            skewness_list_args[i] = skewness_list[i];
            kurtosis_list_args[i] = kurtosis_list[i];
        }

        /* Variables definition */

        acc_weight = IntVarArray(*this, n_item + 1);
        take_item = IntVarArray(*this, n_item);
        total_profit = FloatVar(*this,  -10000000, 10000000);

        for (int i = 0; i < n_item + 1 ; i++) {
            acc_weight[i] = IntVar(*this, 0, 10000000);
        }

        for (int i = 0; i < n_item; i++) {
            take_item[i] = IntVar(*this, 0, 1);
        }

        /* Constraints */
        set_initial_state();

        for (int i = 0; i < n_item  ; i++) {
            validity_condition(i);
            acc_weight[i+1] = transition_weight(i);
        }

        if(opt.discrete_coeffs) {
            cost_discrete_constraint();
        }
        else {
            cost_continuous_constraint();
        }


        switch (opt.model()) {

            case RL_BAB:
                branch(*this, take_item, INT_VAR_NONE(), INT_VAL(&value_selector));
                branch(*this, total_profit, FLOAT_VAL_SPLIT_MAX());
                break;

            case RL_DQN:
                branch(*this, take_item, INT_VAR_NONE(), INT_VAL(&value_selector));
                branch(*this, total_profit, FLOAT_VAL_SPLIT_MAX());
                break;
            case RL_PPO:
                branch(*this, take_item, INT_VAR_NONE(), INT_VAL(&probability_selector));
                branch(*this, total_profit, FLOAT_VAL_SPLIT_MAX());
                break;
            default:
                cout << "Search strategy not implemented" << endl;
                break;

        }
    }

    static int value_selector(const Space& home, IntVar x, int i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        const Portfolio_DP& portfolio_dp = static_cast<const Portfolio_DP&>(home);
        int cur_weight = portfolio_dp.acc_weight[i].val();


        std::vector<double> q_values;
        std::tuple<int, int> key = std::make_tuple(i, cur_weight);
        if (portfolio_dp.use_cache && predict_cache.find(key) != predict_cache.end()){
            q_values = predict_cache[key];
        }
        else{
            py::list y_predict = portfolio_dp.to_run.attr("predict_dqn")(i, cur_weight);
            q_values = y_predict.cast<std::vector<double>>();
            if(portfolio_dp.use_cache){
              predict_cache[key] = q_values;
            }
        }

        int best_q_value = non_fixed_arg_max(2, q_values, x);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        return best_q_value;
    }

    static int probability_selector(const Space& home, IntVar x, int i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        const Portfolio_DP& portfolio_dp = static_cast<const Portfolio_DP&>(home);

        py::list domain_python = py::list();

        //loop on the index of the cities and check if the value is in the set var
        for(IntVarValues it(portfolio_dp.take_item[i]); it(); ++it){
                domain_python.append(it.val());
        }
        int cur_weight = portfolio_dp.acc_weight[i].val();

        std::vector<double> prob_values;
        std::tuple<int, int> key = std::make_tuple(i, cur_weight);
        if (portfolio_dp.use_cache && predict_cache.find(key) != predict_cache.end()){
            prob_values = predict_cache[key];

        }
        else{
            py::list y_predict = portfolio_dp.to_run.attr("predict_ppo")(i, cur_weight, domain_python, portfolio_dp.temperature);
            prob_values = y_predict.cast<std::vector<double>>();
            if(portfolio_dp.use_cache){
              predict_cache[key] = prob_values;
            }
        }

        std::discrete_distribution<int> distribution(prob_values.begin(), prob_values.end());
        int action = distribution(generator);

        return action;

    }



    /* Copy constructor */
    Portfolio_DP(Portfolio_DP& s): FloatMaximizeScript(s) {

        acc_weight.update(*this, s.acc_weight);
        take_item.update(*this, s.take_item);
        total_profit.update(*this, s.total_profit);

        this->n_item = s.n_item;
        this->lambda_1 = s.lambda_1;
        this->lambda_2 = s.lambda_2;
        this->lambda_3 = s.lambda_3;
        this->lambda_4 = s.lambda_4;
        this->temperature = s.temperature;
        this->to_run = s.to_run;


        this->weight_list = s.weight_list;
        this->mean_list = s.mean_list;
        this->deviation_list = s.deviation_list;
        this->skewness_list = s.skewness_list;
        this->kurtosis_list = s.kurtosis_list;

        this->capacity = s.capacity;
        this->seed = s.seed;
        this->use_cache = s.use_cache;
    }

    virtual Portfolio_DP* copy(void) {
        return new Portfolio_DP(*this);
    }

    virtual FloatVar cost(void) const {
        return this->total_profit;
    }

    void set_initial_state() {
        rel(*this, acc_weight[0] == 0);

    }

    void cost_discrete_constraint() {
        IntVar tmp = IntVar(*this, -10000000, 10000000);
        rel(*this, tmp == this->lambda_1 * sum(mean_list, take_item) -
                                   this->lambda_2 * nroot(sum(deviation_list, take_item), 2) +
                                   this->lambda_3 * nroot(sum(skewness_list, take_item), 3) -
                                   this->lambda_4 * nroot(sum(kurtosis_list, take_item), 4));

        channel(*this, tmp, total_profit);

    }

    void cost_continuous_constraint() {
        FloatVarArray tmp(*this, n_item);

        for (int i = 0; i < n_item ; i++) {
            tmp[i] = FloatVar(*this, -10000000, 10000000);
        }


        IntArgs weight_list_args(n_item);
        FloatValArgs mean_list_args(n_item);
        FloatValArgs deviation_list_args(n_item);
        FloatValArgs skewness_list_args(n_item);
        FloatValArgs kurtosis_list_args(n_item);


        for (int i = 0; i <  n_item; i++) {
            weight_list_args[i] = weight_list[i];
            mean_list_args[i] = mean_list[i];
            deviation_list_args[i] = deviation_list[i];
            skewness_list_args[i] = skewness_list[i];
            kurtosis_list_args[i] = kurtosis_list[i];
        }

        rel(*this, total_profit == this->lambda_1 * sum(mean_list_args, tmp) -
                                   this->lambda_2 * nroot(sum(deviation_list_args, tmp), 2) +
                                   this->lambda_3 * nroot(sum(skewness_list_args, tmp), 3) -
                                   this->lambda_4 * nroot(sum(kurtosis_list_args, tmp), 4));

        for (int i = 0; i < n_item; i++) {
            channel(*this, tmp[i], take_item[i]);
        }


    }

    IntVar transition_weight(int stage) {

        IntVar tmp = expr(*this, acc_weight[stage] + take_item[stage] * weight_list[stage]);
        return tmp;
    }

    void validity_condition(int stage) {

        rel(*this, transition_weight(stage) <= capacity);
    }

    virtual void print(std::ostream& os) const {
        std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
        int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds> (end-start).count();

          if( this->cost().val() > best_cost) {
              os << this->n_item << "," << this->seed << "," << elapsed_seconds << ","
                 << this->use_cache << "," << this->cost() << "," << this->take_item  << endl;
          }

    }

    std::string to_string()  {
        std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
        int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds> (end-start).count();

        std::stringstream output;

        output << this->n_item << ","
               << this->seed << ","
               << elapsed_seconds << ","
               << this->use_cache << ","
               << this->cost().min() << ","
               << this->take_item;

        return output.str();
    }

    static int non_fixed_arg_max(int n, std::vector<double> qvalues, IntVar cur_var) {
        int pos = -1;
        double best = -10000000;
        for (int i = 0; i < n; ++i)
            if ((pos == -1 || qvalues[i] > best) && cur_var.in(i))
            {
                pos = i;
                best = qvalues[i];
            }
        assert(pos != -1);
        return pos;
    }

};



Portfolio_DP::ModelType stringToModel (std::string const& inString) {
    if (inString == "rl-bab-dqn") return Portfolio_DP::RL_BAB;
    if (inString == "rl-ilds-dqn") return Portfolio_DP::RL_DQN;
    if (inString == "rl-rbs-ppo") return Portfolio_DP::RL_PPO;
    throw cxxopts::OptionException("Invalid model argument");
}



int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};

    cxxopts::Options options("PortfolioSolver", "Portfolio solver for DQN and PPO");
    options.add_options()
    ("luby", "luby scaling factor", cxxopts::value<int>()->default_value("1"))
    ("temperature", "temperature for the randomness", cxxopts::value<float>()->default_value("1.0"))
    ("size", "instance size", cxxopts::value<int>()->default_value("50"))
    ("capacity_ratio", "capacity ratio", cxxopts::value<float>()->default_value("0.5"))
    ("lambda_1", "mean factor", cxxopts::value<int>()->default_value("1"))
    ("lambda_2", "deviation factor", cxxopts::value<int>()->default_value("5"))
    ("lambda_3", "skewness factor", cxxopts::value<int>()->default_value("5"))
    ("lambda_4", "kurtosis factor", cxxopts::value<int>()->default_value("5"))
    ("seed", "random seed", cxxopts::value<int>()->default_value("19"))
    ("time", "Time limit in ms", cxxopts::value<int>()->default_value("6000"))
    ("d_l", "LDS cutoff", cxxopts::value<int>()->default_value("50"))
    ("model", "model to run", cxxopts::value<string>())
    ("cache", "enable cache", cxxopts::value<bool>()->default_value("1"))
    ("discrete_coeffs", "floor the sqrt values", cxxopts::value<bool>()->default_value("0"));
    auto result = options.parse(argc, argv);

    OptionsPortfolio opt("Portfolio problem",
                        result["size"].as<int>(),
                        result["cache"].as<bool>(),
                        result["capacity_ratio"].as<float>(),
                        result["lambda_1"].as<int>(),
                        result["lambda_2"].as<int>(),
                        result["lambda_3"].as<int>(),
                        result["lambda_4"].as<int>(),
                        result["temperature"].as<float>(),
                        result["discrete_coeffs"].as<bool>()
                        );
    opt.model(Portfolio_DP::RL_BAB, "rl-bab-dqn", "use RL with BAB and DQN");
    opt.model(Portfolio_DP::RL_DQN, "rl-ilds-dqn", "use RL with ILDS and DQN");
    opt.model(Portfolio_DP::RL_PPO, "rl-rbs-ppo", "use RL with RBS and PPO (implies restarts)");
    // TODO switch case on the model?
    opt.model(stringToModel(result["model"].as<string>()));

    opt.solutions(0); // Find best solution by default
    opt.seed(result["seed"].as<int>());
    opt.time(result["time"].as<int>());
    opt.d_l(result["d_l"].as<int>());

    Portfolio_DP* p = new Portfolio_DP(opt);

    if(opt.model() == Portfolio_DP::RL_DQN) {
        Search::Options o;
        Search::TimeStop ts(opt.time());
        o.stop = &ts;
        o.d_l = opt.d_l();

        Search::Cutoff* c = Search::Cutoff::luby(result["luby"].as<int>());
        o.cutoff = c;

        LDS<Portfolio_DP> engine(p,o);
        delete p;

        cout << "nb_cities,seed,time,portfolio_profit,solution" << std::endl;
        while(Portfolio_DP* p = engine.next()) {
            FloatVal cur_cost = p->cost().val();
            if(cur_cost > best_cost) {
                best_cost = cur_cost;
                int num_nodes = -1;
                int num_fail = -1;
                cout << p->to_string() << endl;
            }
            delete p;
        }
        cout << "BEST SOLUTION: " << best_cost << endl;
        if(engine.stopped()){
            cout << "TIMEOUT - OPTIMALITY PROOF NOT REACHED" << endl;
        }
        else{
            cout << "SEARCH COMPLETED - SOLUTION FOUND IS OPTIMAL" << endl;
        }
    }


     else if(opt.model() == Portfolio_DP::RL_BAB) {
        Search::Options o;
        Search::TimeStop ts(opt.time());
        o.stop = &ts;
        o.d_l = opt.d_l();


        Search::Cutoff* c = Search::Cutoff::luby(result["luby"].as<int>());
        o.cutoff = c;

        BAB<Portfolio_DP> engine(p,o);

        delete p;

        cout << "nb_cities,seed,time,portfolio_profit,solution" << std::endl;
        while(Portfolio_DP* p = engine.next()) {
            FloatVal cur_cost = p->cost().val();
            if(cur_cost > best_cost) {
                best_cost = cur_cost;
                int num_nodes = -1;
                int num_fail = -1;
                cout << p->to_string() << endl;
            }
            delete p;
        }
        cout << "BEST SOLUTION: " << best_cost << endl;
        if(engine.stopped()){
            cout << "TIMEOUT - OPTIMALITY PROOF NOT REACHED" << endl;
        }
        else{
            cout << "SEARCH COMPLETED - SOLUTION FOUND IS OPTIMAL" << endl;
        }
    }

    else if(opt.model() == Portfolio_DP::RL_PPO)  {
        Search::Options o;
        Search::TimeStop ts(opt.time());
        o.stop = &ts;

        Search::Cutoff* c = Search::Cutoff::luby(result["luby"].as<int>());
        o.cutoff = c;

        RBS<Portfolio_DP,BAB> engine(p,o);
        delete p;

        cout << "nb_cities,seed,time,portfolio_profit,solution" << std::endl;
        while(Portfolio_DP* p = engine.next()) {
            FloatVal cur_cost = p->cost().val();
            if(cur_cost > best_cost) {
                best_cost = cur_cost;
                int num_nodes = engine.statistics().node;
                int num_fail = engine.statistics().fail;
                cout << p->to_string() << endl;
            }
            delete p;
        }
         cout << "BEST SOLUTION: " << best_cost << endl;
        if(engine.stopped()){
            cout << "TIMEOUT - OPTIMALITY PROOF NOT REACHED" << endl;
        }
        else{
            cout << "SEARCH COMPLETED - SOLUTION FOUND IS OPTIMAL" << endl;
        }
    }

    else {
        cout << "Model not implemented" << std::endl;
    }

    return 0;
}

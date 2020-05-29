#include <chrono>
#include <random>
#include <math.h>
#include <ctime>

#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/set.hh>
#include <gecode/search.hh>
#include <gecode/minimodel.hh>


#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <map>
#include <limits>
#include <string>
#include <cxxopts.hpp>

using namespace Gecode;
using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

std::chrono::time_point<std::chrono::system_clock> start;
std::map<std::tuple<std::vector<int>, int>, std::vector<double>> predict_cache;

Rnd r(1U);
auto t = time(0);
std::default_random_engine generator(0);

int best_cost = 100000000;


class OptionsTSPTW: public Options {
public:
  int number_cities;
  int grid_size;
  int max_tw_gap;
  int max_tw_size;
  bool cache;
  float temperature;
  bool dominance;
  /// Initialize options for example with name \a s
  OptionsTSPTW(const char* s, int number_cities0, int grid_size0, int max_tw_gap0, int max_tw_size0,
              bool cache0,  float temperature0, bool dominance0)
    : Options(s), number_cities(number_cities0), grid_size(grid_size0), max_tw_gap(max_tw_gap0), max_tw_size(max_tw_size0),
      cache(cache0), temperature(temperature0), dominance(dominance0) {}
};

class TSPTW_DP : public IntMinimizeScript {
private:

    static const int max_distance = 150;
    static const int max_time = 10000;
    IntVar tour_cost;
    SetVarArray must_visit_city;
    IntVarArray last_city;
    IntVarArray travel_to;
    IntVarArray acc_cost;
    IntVarArray time;
    int m;
    float temperature;
    py::object python_binding;
    std::vector<std::vector<double>> travel_time;
    std::vector<std::vector<double>> time_windows;
    bool use_cache;
    int seed;

public:

    enum ModelType {RL_BAB, RL_DQN, RL_PPO, NN_HEURISTIC};

    /* Constructor */
    TSPTW_DP(const OptionsTSPTW& opt): IntMinimizeScript(opt), tour_cost(*this, 0, 100000) {

        m = opt.number_cities;
        seed = opt.seed();
        this->temperature = opt.temperature;
        this->use_cache = opt.cache;

        start = std::chrono::system_clock::now();

        string mode = "";

        if (opt.model() == RL_DQN || opt.model() == RL_BAB || opt.model() == NN_HEURISTIC) {
            mode = "dqn";
        }

        else if(opt.model() == RL_PPO) {
            mode = "ppo";
        }

        string model_folder = "./selected-models/" + mode + "/tsptw/n-city-" + std::to_string(m) +
                              "/grid-" + std::to_string(opt.grid_size) + "-tw-" + std::to_string(opt.max_tw_gap)
                              + "-" + std::to_string(opt.max_tw_size);

        auto to_run_module = py::module::import("src.problem.tsptw.solving.solver_binding");
        python_binding = to_run_module.attr("SolverBinding")(model_folder, m, opt.grid_size, opt.max_tw_gap, opt.max_tw_size, seed, mode);


        auto travel_time_python = python_binding.attr("get_travel_time_matrix")();
        this->travel_time = travel_time_python.cast<std::vector<std::vector<double>>>();

        auto time_windows_python = python_binding.attr("get_time_windows_matrix")();
        this->time_windows = time_windows_python.cast<std::vector<std::vector<double>>>();

        /* Parameters */
        IntArgs e(m * m);
        Matrix<IntArgs> travel_time_matrix(e, m);

        for (int i = 0; i <  m; i++) {
            for(int j = 0; j < m; j++) {
            travel_time_matrix(i,j) = travel_time[i][j];
            }

        }

        IntArgs etw(m * 2);
        Matrix<IntArgs> time_windows_matrix(etw, m, 2);

        for (int i = 0; i <  m; i++) {
            time_windows_matrix(i, 0) = time_windows[i][0];
            time_windows_matrix(i, 1) = time_windows[i][1];

        }

        /* Variables definition */
        must_visit_city = SetVarArray(*this, m);
        last_city = IntVarArray(*this, m);
        travel_to = IntVarArray(*this, m - 1);
        acc_cost = IntVarArray(*this, m);
        time = IntVarArray(*this, m);

        for (int i = 0; i < m ; i++) {
            must_visit_city[i] = SetVar(*this, IntSet::empty, IntSet(1, m - 1));
            last_city[i] = IntVar(*this, 0, m - 1);

            acc_cost[i] = IntVar(*this, 0, this->max_distance);
            time[i] = IntVar(*this, 0, this->max_time);
        }

        for (int i = 0; i < m - 1; i++) {
            travel_to[i] = IntVar(*this, 1, m - 1);
        }

        /* Constraints */
        set_initial_state();

        for (int i = 0; i < m - 1 ; i++) {
            validity_condition(i, travel_time_matrix, time_windows_matrix);

            must_visit_city[i+1] = transition_visited(i);
            last_city[i+1] = transition_last_city(i);
            acc_cost[i+1] = transition_cost(i, travel_time_matrix);
            time[i+1] = transition_time(i, travel_time_matrix, time_windows_matrix);

            if (opt.dominance) {
                dominance_pruning(i, time_windows_matrix);
            }
        }

        /* objective function */
        IntVar return_cost = IntVar(*this, 0, this->max_distance);
        element(*this, travel_time_matrix, last_city[m-1], IntVar(*this, 0, 0), return_cost);

        rel(*this, tour_cost == acc_cost[m-1] + return_cost);

        switch (opt.model()) {

            case RL_DQN:
                branch(*this, travel_to, INT_VAR_NONE(), INT_VAL(&value_selector));
                break;
            case RL_BAB:
                branch(*this, travel_to, INT_VAR_NONE(), INT_VAL(&value_selector));
                break;
            case NN_HEURISTIC:
                branch(*this, travel_to, INT_VAR_NONE(), INT_VAL(&nearest_neighbour_value_selector));
                break;
            case RL_PPO:
                branch(*this, travel_to, INT_VAR_NONE(), INT_VAL(&probability_selector));
                break;
            default:
                cout << "Search strategy not implemented" << endl;
                break;
        }
    }

    static int value_selector(const Space& home, IntVar x, int i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        const TSPTW_DP& tsptw_dp = static_cast<const TSPTW_DP&>(home);

        py::list non_fixed_variables_python = py::list();
        std::vector<int> non_fixed_variables;

        //loop on the index of the cities and check if the value is in the set var
        for(int index=0; index < tsptw_dp.must_visit_city.size(); index++){
            if(tsptw_dp.must_visit_city[i].contains(index)){
                non_fixed_variables_python.append(index);
                non_fixed_variables.push_back(index);
            }
        }
        int last_visited = tsptw_dp.last_city[i].val();


        std::vector<double> q_values;
        std::tuple<std::vector<int>, int> key = std::make_tuple(non_fixed_variables, last_visited);
        if (tsptw_dp.use_cache && predict_cache.find(key) != predict_cache.end()){
            q_values = predict_cache[key];
        }
        else{
            py::list y_predict = tsptw_dp.python_binding.attr("predict_dqn")(non_fixed_variables, last_visited);
            q_values = y_predict.cast<std::vector<double>>();
            if(tsptw_dp.use_cache){
              predict_cache[key] = q_values;
            }
        }

        int best_q_value = non_fixed_arg_max(tsptw_dp.m, q_values, x);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

        return best_q_value;
    }

    static int probability_selector(const Space& home, IntVar x, int i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        const TSPTW_DP& tsptw_dp = static_cast<const TSPTW_DP&>(home);

        py::list non_fixed_variables_python = py::list();
        std::vector<int> non_fixed_variables;
        for(int index=0; index < tsptw_dp.must_visit_city.size(); index++){
            if(tsptw_dp.must_visit_city[i].contains(index)){
                non_fixed_variables_python.append(index);
                non_fixed_variables.push_back(index);
            }
        }
        int last_visited = tsptw_dp.last_city[i].val();


        std::vector<double> prob_values;
        std::tuple<std::vector<int>, int> key = std::make_tuple(non_fixed_variables, last_visited);

        if (tsptw_dp.use_cache && predict_cache.find(key) != predict_cache.end()) {
            prob_values = predict_cache[key];
        }
        else {
            py::list y_predict = tsptw_dp.python_binding.attr("predict_ppo")(non_fixed_variables, last_visited, tsptw_dp.temperature);
            prob_values = y_predict.cast<std::vector<double>>();

            if(tsptw_dp.use_cache) {
              predict_cache[key] = prob_values;
            }
        }

        std::discrete_distribution<int> distribution(prob_values.begin(), prob_values.end());
        int action = distribution(generator);

        return action;

    }

     static int nearest_neighbour_value_selector(const Space& home, IntVar x, int i) {
        const TSPTW_DP& tsptw_dp = static_cast<const TSPTW_DP&>(home);
        int last_visited = tsptw_dp.last_city[i].val();
        int best_q_value = get_nearest_non_visited(tsptw_dp.m, last_visited, tsptw_dp.travel_time, x);
        return best_q_value;
    }

    static int get_nearest_non_visited(int n, int last_visited,
                                       std::vector<std::vector<double>> travel_time, IntVar cur_var) {
        //set the visited cities distance to infinity then return index of min value (arg_min)
        int infinity = std::numeric_limits<int>::max();
        int best = infinity;
        int best_index = -1;
        for (int i = 0; i < n; ++i){
            if (travel_time[last_visited][i] < best && cur_var.in(i)){
                best = travel_time[last_visited][i];
                best_index = i;
            }
        }

        assert(best_index >= 0);
        return best_index;
    }

    /* Copy constructor */
    TSPTW_DP(TSPTW_DP& s): IntMinimizeScript(s) {
        tour_cost.update(*this, s.tour_cost);
        must_visit_city.update(*this, s.must_visit_city);
        last_city.update(*this, s.last_city);
        travel_to.update(*this, s.travel_to);
        acc_cost.update(*this, s.acc_cost);
        time.update(*this, s.time);
        this->m = s.m;
        this->temperature = s.temperature;
        this->python_binding = s.python_binding;
        this->travel_time = s.travel_time;
        this->seed = s.seed;
        this->use_cache = s.use_cache;
    }

    virtual TSPTW_DP* copy(void) {
        return new TSPTW_DP(*this);
    }

    virtual IntVar cost(void) const {

        return tour_cost;
    }


    void set_initial_state() {
        rel(*this, must_visit_city[0] == IntSet(1, this->m - 1));
        rel(*this, last_city[0] == 0);
        rel(*this, acc_cost[0] == 0);
        rel(*this, time[0] == 0);
    }

    SetVar transition_visited(int stage) {
        SetVar tmp = expr(*this, must_visit_city[stage] - singleton(travel_to[stage])); // - stand for the difference
        return tmp;
    }

    IntVar transition_last_city(int stage) {
        return travel_to[stage];
    }

    IntVar transition_time(int stage, Matrix<IntArgs> travel_time_matrix, Matrix<IntArgs> time_windows_matrix) {

        IntVar tmp_travel = IntVar(*this, 0, this->max_distance);
        IntVar tmp_time = IntVar(*this, 0, this->max_time);

        element(*this, travel_time_matrix, last_city[stage], travel_to[stage], tmp_travel);
        element(*this, time_windows_matrix, travel_to[stage], IntVar(*this, 0, 0), tmp_time);

        IntVar tmp_expr_travel = expr(*this, time[stage] + tmp_travel);
        IntVar tmp_ret = IntVar(*this, 0, this->max_time);

        max(*this, tmp_expr_travel, tmp_time, tmp_ret);

        return tmp_ret;
    }

    IntVar transition_cost(int stage, Matrix<IntArgs> travel_time_matrix) {

        IntVar tmp = IntVar(*this, 0, this->max_distance);
        element(*this, travel_time_matrix, last_city[stage], travel_to[stage], tmp);

        return expr(*this, acc_cost[stage] + tmp);
    }

    void validity_condition(int stage, Matrix<IntArgs> travel_time_matrix, Matrix<IntArgs> time_windows_matrix) {

         // cannot visit a city already visited
        rel(*this,  (must_visit_city[stage] >= singleton(travel_to[stage])));

        // Visit must be inside the time windows
        IntVar tmp_time_lb = IntVar(*this, 0, this->max_time);
        IntVar tmp_time_ub = IntVar(*this, 0, this->max_time);

        element(*this, time_windows_matrix, travel_to[stage], IntVar(*this, 0, 0), tmp_time_lb);
        element(*this, time_windows_matrix, travel_to[stage], IntVar(*this, 1, 1), tmp_time_ub);

        rel(*this, transition_time(stage, travel_time_matrix, time_windows_matrix) >= tmp_time_lb);
        rel(*this, transition_time(stage, travel_time_matrix, time_windows_matrix) <= tmp_time_ub);
    }

    void dominance_pruning(int stage, Matrix<IntArgs> time_windows_matrix) {

        for (int i = 0; i < m  ; i++) {
            BoolVar tmp_lhs = BoolVar(*this, 0, 1);
            BoolVar tmp_rhs = BoolVar(*this, 0, 1);
            BoolVar tmp_fin = BoolVar(*this, 1, 1);
            IntVar tmp_elem = IntVar(*this, 0, this->max_time);

            element(*this, time_windows_matrix, IntVar(*this, i, i), IntVar(*this, 1, 1), tmp_elem);

            rel(*this,  tmp_elem, IRT_LQ, time[stage], tmp_lhs);

            rel(*this, must_visit_city[stage], SRT_SUP, IntVar(*this, i, i), tmp_rhs);

            Reify r(tmp_lhs, RM_IMP);
            rel(*this, tmp_fin, IRT_NQ, tmp_rhs, r);
        }

    }

    static int non_fixed_arg_max(int n, std::vector<double> q_values, IntVar cur_var) {
        int pos = -1;
        double best = -10000000;
        for (int i = 0; i < n; ++i)
            if ((pos == -1 || q_values[i] > best) && cur_var.in(i))
            {
                pos = i;
                best = q_values[i];
            }
        assert(pos != -1);
        return pos;
    }

    std::string to_string()  {

        std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
        int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds> (end-start).count();
        std::stringstream output;
        output << this->m << "," << this->seed << "," << elapsed_seconds << "," << this->tour_cost << "," << this->travel_to;

        return output.str();
    }

};


TSPTW_DP::ModelType stringToModel (std::string const& inString) {
    if (inString == "rl-bab-dqn") return TSPTW_DP::RL_BAB;
    if (inString == "rl-ilds-dqn") return TSPTW_DP::RL_DQN;
    if (inString == "rl-rbs-ppo") return TSPTW_DP::RL_PPO;
    if (inString == "nearest") return TSPTW_DP::NN_HEURISTIC;
    throw cxxopts::OptionException("Invalid model argument");
}


int main(int argc, char* argv[]) {
    /* Python embedding */
    py::scoped_interpreter guard{};

    cxxopts::Options options("PybindGecode", "One line description of MyProgram");
    options.add_options()
    ("luby", "luby scaling factor", cxxopts::value<int>()->default_value("1"))
    ("temperature", "temperature for the randomness", cxxopts::value<float>()->default_value("1.0"))
    ("size", "instance size", cxxopts::value<int>()->default_value("50"))
    ("grid_size", "maximum grid size for generating the cities", cxxopts::value<int>()->default_value("100"))
    ("max_tw_gap", "maximum gap between two consecutive time windows in the feasible solution generated", cxxopts::value<int>()->default_value("10"))
    ("max_tw_size", "maximum time windows size", cxxopts::value<int>()->default_value("100"))
    ("seed", "random seed", cxxopts::value<int>()->default_value("19"))
    ("time", "Time limit in ms", cxxopts::value<int>()->default_value("6000"))
    ("d_l", "LDS cutoff", cxxopts::value<int>()->default_value("50"))
    ("model", "model to run", cxxopts::value<string>())
    ("dominance", "enable dominance pruning", cxxopts::value<bool>()->default_value("1"))
    ("cache", "enable cache", cxxopts::value<bool>()->default_value("1"));
    auto result = options.parse(argc, argv);

    OptionsTSPTW opt("TSPTW problem",
                        result["size"].as<int>(),
                        result["grid_size"].as<int>(),
                        result["max_tw_gap"].as<int>(),
                        result["max_tw_size"].as<int>(),
                        result["cache"].as<bool>(),
                        result["temperature"].as<float>(),
                        result["dominance"].as<bool>()
                        );
    opt.model(TSPTW_DP::RL_BAB, "rl-bab-dqn", "use RL with BAB and DQN");
    opt.model(TSPTW_DP::RL_DQN, "rl-ilds-dqn", "use RL with ILDS and DQN");
    opt.model(TSPTW_DP::RL_PPO, "rl-rbs-ppo", "use RL with RBS and PPO (implies restarts)");
    opt.model(TSPTW_DP::NN_HEURISTIC, "nearest", "use nearest heuristic with ILDS");
    // TODO switch case on the model?
    opt.model(stringToModel(result["model"].as<string>()));

    opt.solutions(0);
    opt.seed(result["seed"].as<int>());
    opt.time(result["time"].as<int>());
    opt.d_l(result["d_l"].as<int>());


    if(opt.model() == TSPTW_DP::RL_DQN || opt.model() == TSPTW_DP::NN_HEURISTIC) {
        Search::Options o;
        Search::TimeStop ts(opt.time());
        o.stop = &ts;
        TSPTW_DP* p = new TSPTW_DP(opt);
        o.d_l = opt.d_l();
        LDS<TSPTW_DP> engine(p, o);
        delete p;
        cout << "n_city,seed,time,tour_cost" << std::endl;
        while(TSPTW_DP* p = engine.next()) {
            int cur_cost = p->cost().val();
            if(cur_cost < best_cost) {
                best_cost = cur_cost;
                int depth = engine.statistics().depth;
                cout << p->to_string()  << endl ;
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


     else if(opt.model() == TSPTW_DP::RL_BAB) {
        Search::Options o;
        Search::TimeStop ts(opt.time());
        o.stop = &ts;
        TSPTW_DP* p = new TSPTW_DP(opt);
        o.d_l = opt.d_l();
        BAB<TSPTW_DP> engine(p, o);
        delete p;
        cout << "n_city,seed,time,tour_cost" << std::endl;
        while(TSPTW_DP* p = engine.next()) {
            int cur_cost = p->cost().val();
            if(cur_cost < best_cost) {
                best_cost = cur_cost;
                int depth = engine.statistics().depth;
                cout << p->to_string() << endl ;
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


    else if(opt.model() == TSPTW_DP::RL_PPO) {
        Search::Options o;

        Search::TimeStop ts(opt.time());
        o.stop = &ts;
        TSPTW_DP* p = new TSPTW_DP(opt);

        Search::Cutoff* c = Search::Cutoff::luby(result["luby"].as<int>());
        o.cutoff = c;
        RBS<TSPTW_DP,BAB> engine(p,o);
        delete p;

        cout << "n_city,seed,time,tour_cost" << std::endl;
        while(TSPTW_DP* p = engine.next()) {
            int cur_cost = p->cost().val();
            if(cur_cost < best_cost) {
                best_cost = cur_cost;
                int num_nodes = engine.statistics().node;
                int num_fail = engine.statistics().fail;
                cout << p->to_string() << endl ;
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



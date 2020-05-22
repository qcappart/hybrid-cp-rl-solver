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


class OptionsPDTSP: public Options {
public:
  int number_cities;
  int number_commodities;
  int grid_size;
  float max_commodity_weight;
  float capacity_looseness;
  bool cache;
  float temperature;
  bool dominance;
  /// Initialize options for example with name \a s
  OptionsPDTSP(const char* s, int number_cities0, int number_commodities0, int grid_size0, float max_commodity_weight0, float capacity_looseness0,
              bool cache0,  float temperature0, bool dominance0)
    : Options(s), number_cities(number_cities0), number_commodities(number_commodities0), grid_size(grid_size0), max_commodity_weight(max_commodity_weight0), capacity_looseness(capacity_looseness0),
      cache(cache0), temperature(temperature0), dominance(dominance0) {}
};

class PDTSP_DP : public IntMinimizeScript {
private:

    static const int max_distance = 150;
    static const int max_load = 10000;
    IntVar tour_cost;
    SetVarArray must_visit_city;
    IntVarArray last_city;
    IntVarArray travel_to;
    IntVarArray acc_cost;
    IntVarArray load;
    int n;
    int m;
    float temperature;
    py::object python_binding;
    std::vector<std::vector<double>> distances;
    std::vector<std::vector<double>> pickup_constraints;
    int capacity;
    bool use_cache;
    int seed;

public:

    enum ModelType {RL_DQN, RL_PPO, NN_HEURISTIC};

    /* Constructor */
    PDTSP_DP(const OptionsPDTSP& opt): IntMinimizeScript(opt), tour_cost(*this, 0, 100000) {

        n = opt.number_cities;
        m = opt.number_commodities;
        seed = opt.seed();
        this->temperature = opt.temperature;
        this->use_cache = opt.cache;

        start = std::chrono::system_clock::now();

        string mode = "";

        if (opt.model() == RL_DQN || opt.model() == NN_HEURISTIC) {
            mode = "dqn";
        }

        else if(opt.model() == RL_PPO) {
            mode = "ppo";
        }

        string model_folder = "./selected-models/" + mode + "/pdtsp/n-city-" + std::to_string(n) +
                              "/m_commod-" +std::to_string(m) + "-grid-" + std::to_string(opt.grid_size) + "-cw-" + std::to_string(opt.max_commodity_weight)
                              + "-" + std::to_string(opt.capacity_looseness);

        auto to_run_module = py::module::import("src.problem.pdtsp.solving.solver_binding");
        python_binding = to_run_module.attr("SolverBinding")(model_folder, n, m, opt.grid_size, opt.max_commodity_weight, opt.capacity_looseness, seed, mode);


        auto distances_python = python_binding.attr("get_distances_matrix")();
        this->distances = distances_python.cast<std::vector<std::vector<double>>>();

        auto pickup_constraints_python = python_binding.attr("get_pickup_constraints_matrix")();
        this->pickup_constraints = pickup_constraints_python.cast<std::vector<std::vector<double>>>();

        auto capacity_python = python_binding.attr("get_capacity")();
        this->capacity = capacity_python.cast<int>();

        /* Parameters */
        IntArgs e(n * n);
        Matrix<IntArgs> distances_matrix(e, n);

        for (int i = 0; i <  n; i++) {
            for(int j = 0; j < n; j++) {
            distances_matrix(i,j) = distances[i][j];
            }

        }

        IntArgs epc(n * m);
        Matrix<IntArgs> pickup_constraints_matrix(etw, n, m);

        for (int i = 0; i <  n; i++) {
            for (int j = ; j < m; j++) {
                pickup_constraints_matrix(i, j) = pickup_constraints[i][j];
            }
        }

        /* Variables definition */
        must_visit_city = SetVarArray(*this, n);
        last_city = IntVarArray(*this, n);
        travel_to = IntVarArray(*this, n - 1);
        acc_cost = IntVarArray(*this, n);
        
        load = IntVarArray(*this, n * m);

        for (int i = 0; i < n ; i++) {
            must_visit_city[i] = SetVar(*this, IntSet::empty, IntSet(1, n - 1));
            last_city[i] = IntVar(*this, 0, n - 1);

            acc_cost[i] = IntVar(*this, 0, this->max_distance);
            for (int j = 0; j < m; j++) {
                load[j + m * i] = IntVar(*this, 0, this->max_load);
            }
        }

        for (int i = 0; i < n - 1; i++) {
            travel_to[i] = IntVar(*this, 1, n - 1);
        }

        /* Constraints */
        set_initial_state();

        for (int i = 0; i < n - 1 ; i++) {
            validity_condition(i, distances_matrix, pickup_constraints_matrix);

            must_visit_city[i+1] = transition_visited(i);
            last_city[i+1] = transition_last_city(i);
            acc_cost[i+1] = transition_cost(i, distances_matrix);            
            for (int j = 0; j < m; j++) {
                load[i*(m+1) + j] = transition_load(i, j, pickup_constraints_matrix);
            }
            if (opt.dominance) {
                dominance_pruning(i, pickup_constraints_matrix);
            }
        }


        /* objective function */
        IntVar return_cost = IntVar(*this, 0, this->max_distance);
        element(*this, distances_matrix, last_city[n-1], IntVar(*this, 0, 0), return_cost);

        rel(*this, tour_cost == acc_cost[m-1] + return_cost);

        switch (opt.model()) {

            case RL_DQN:
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
        const PDTSP_DP& pdtsp_dp = static_cast<const PDTSP_DP&>(home);

        py::list non_fixed_variables_python = py::list();
        std::vector<int> non_fixed_variables;

        //loop on the index of the cities and check if the value is in the set var
        for(int index=0; index < pdtsp_dp.must_visit_city.size(); index++){
            if(pdtsp_dp.must_visit_city[i].contains(index)){
                non_fixed_variables_python.append(index);
                non_fixed_variables.push_back(index);
            }
        }
        int last_visited = pdtsp_dp.last_city[i].val();


        std::vector<double> q_values;
        std::tuple<std::vector<int>, int> key = std::make_tuple(non_fixed_variables, last_visited);
        if (pdtsp_dp.use_cache && predict_cache.find(key) != predict_cache.end()){
            q_values = predict_cache[key];
        }
        else{
            py::list y_predict = pdtsp_dp.python_binding.attr("predict_dqn")(non_fixed_variables, last_visited);
            q_values = y_predict.cast<std::vector<double>>();
            if(pdtsp_dp.use_cache){
              predict_cache[key] = q_values;
            }
        }

        int best_q_value = non_fixed_arg_max(pdtsp_dp.m, q_values, x);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

        return best_q_value;
    }

    static int probability_selector(const Space& home, IntVar x, int i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        const PDTSP_DP& pdtsp_dp = static_cast<const PDTSP_DP&>(home);

        py::list non_fixed_variables_python = py::list();
        std::vector<int> non_fixed_variables;
        for(int index=0; index < pdtsp_dp.must_visit_city.size(); index++){
            if(pdtsp_dp.must_visit_city[i].contains(index)){
                non_fixed_variables_python.append(index);
                non_fixed_variables.push_back(index);
            }
        }
        int last_visited = pdtsp_dp.last_city[i].val();


        std::vector<double> prob_values;
        std::tuple<std::vector<int>, int> key = std::make_tuple(non_fixed_variables, last_visited);

        if (pdtsp_dp.use_cache && predict_cache.find(key) != predict_cache.end()) {
            prob_values = predict_cache[key];
        }
        else {
            py::list y_predict = pdtsp_dp.python_binding.attr("predict_ppo")(non_fixed_variables, last_visited, pdtsp_dp.temperature);
            prob_values = y_predict.cast<std::vector<double>>();

            if(pdtsp_dp.use_cache) {
              predict_cache[key] = prob_values;
            }
        }

        std::discrete_distribution<int> distribution(prob_values.begin(), prob_values.end());
        int action = distribution(generator);

        return action;

    }

     static int nearest_neighbour_value_selector(const Space& home, IntVar x, int i) {
        const PDTSP_DP& pdtsp_dp = static_cast<const PDTSP_DP&>(home);
        int last_visited = pdtsp_dp.last_city[i].val();
        int best_q_value = get_nearest_non_visited(pdtsp_dp.m, last_visited, pdtsp_dp.distances, x);
        return best_q_value;
    }

    static int get_nearest_non_visited(int n, int last_visited,
                                       std::vector<std::vector<double>> distances, IntVar cur_var) {
        //set the visited cities distance to infinity then return index of min value (arg_min)
        int infinity = std::numeric_limits<int>::max();
        int best = infinity;
        int best_index = -1;
        for (int i = 0; i < n; ++i){
            if (distances[last_visited][i] < best && cur_var.in(i)){
                best = distances[last_visited][i];
                best_index = i;
            }
        }

        assert(best_index >= 0);
        return best_index;
    }

    /* Copy constructor */
    PDTSP_DP(PDTSP_DP& s): IntMinimizeScript(s) {
        tour_cost.update(*this, s.tour_cost);
        must_visit_city.update(*this, s.must_visit_city);
        last_city.update(*this, s.last_city);
        travel_to.update(*this, s.travel_to);
        acc_cost.update(*this, s.acc_cost);
        time.update(*this, s.time);
        this->m = s.m;
        this->temperature = s.temperature;
        this->python_binding = s.python_binding;
        this->distances = s.distances;
        this->seed = s.seed;
        this->use_cache = s.use_cache;
    }

    virtual PDTSP_DP* copy(void) {
        return new PDTSP_DP(*this);
    }

    virtual IntVar cost(void) const {

        return tour_cost;
    }


    void set_initial_state() {
        rel(*this, must_visit_city[0] == IntSet(1, this->m - 1));
        rel(*this, last_city[0] == 0);
        rel(*this, acc_cost[0] == 0);
        for (int i = 0; i < m; i++) {
            rel(*this, load[i] == 0);
        }
    }

    SetVar transition_visited(int stage) {
        SetVar tmp = expr(*this, must_visit_city[stage] - singleton(travel_to[stage])); // - stand for the difference
        return tmp;
    }

    IntVar transition_last_city(int stage) {
        return travel_to[stage];
    }

    IntVar transition_load(int stage, int j, Matrix<IntArgs> pickup_constraints_matrix) {

        IntVar tmp_load = IntVar(*this, 0, this->max_load);
        IntVar current_load = IntVar(*this, 0, this->max_load);

        element(*this, pickup_constraints_matrix, travel_to[stage], j, tmp_load);
        element(*this, load, stage * m + j, current_load);

        return expr(*this, current_load + tmp_load);
    }

    IntVar transition_cost(int stage, Matrix<IntArgs> distances_matrix) {

        IntVar tmp = IntVar(*this, 0, this->max_distance);
        element(*this, distances_matrix, last_city[stage], travel_to[stage], tmp);

        return expr(*this, acc_cost[stage] + tmp);
    }

    void validity_condition(int stage, Matrix<IntArgs> distances_matrix, Matrix<IntArgs> pickup_constraints_matrix) {

         // cannot visit a city already visited
        rel(*this,  (must_visit_city[stage] >= singleton(travel_to[stage])));

        // load can never be negative for any item
        for (int j = 0; j < m; j++) {
            rel(*this, transition_load(stage, j, pickup_constraints_matrix) >= 0);
        }

        //load cannot exceed capacity
        IntVarArray abs_loads = IntVarArray(*this, m);
        for (int j = 0; j < m; j++) {
            rel(*this, abs_loads[j] = abs(load[stage * m + j]));
        }
        rel(*this, sum(abs_loads) <= this->capacity);

    }

    void dominance_pruning(int stage, Matrix<IntArgs> pickup_constraints_matrix) {

        // load can never be negative for any item
        for (int j = 0; j < m; j++) {
            rel(*this, transition_load(stage, j, pickup_constraints_matrix) >= 0);
        }
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
        output << this->m << "," << this->seed << "," << elapsed_seconds << "," << tour_cost;

        return output.str();
    }

};

class SimpleSearchTracer : public SearchTracer {
protected:
  static const char* t2s(EngineType et)  {
    switch (et) {
    case EngineType::DFS: return "DFS";
    case EngineType::BAB: return "BAB";
    case EngineType::LDS: return "LDS";
    case EngineType::RBS: return "RBS";
    case EngineType::PBS: return "PBS";
    case EngineType::AOE: return "AOE";
    }
  }
public:

  int n_node;

  SimpleSearchTracer(void) {
    this->n_node = 0;
  }
  virtual void init(void) {
  }
  virtual void node(const EdgeInfo& ei, const NodeInfo& ni) {
    this->n_node  = this->n_node + 1;
  }
  virtual void round(unsigned int eid) {
    //std::cout << "accumulated number of nodes: " << this->n_node  << std::endl;
  }
  virtual void skip(const EdgeInfo& ei) {

  }
  virtual void done(void) {
    std::cout << "total node: " << this->n_node  << std::endl;
  }
  virtual ~SimpleSearchTracer(void) {}
};


PDTSP_DP::ModelType stringToModel (std::string const& inString) {
    if (inString == "rl-dqn") return PDTSP_DP::RL_DQN;
    if (inString == "rl-ppo") return PDTSP_DP::RL_PPO;
    if (inString == "nearest") return PDTSP_DP::NN_HEURISTIC;
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
    ("m_commodities", "number of commodities", cxxopts::value<int>()->default_value("10"))
    ("grid_size", "maximum grid size for generating the cities", cxxopts::value<int>()->default_value("100"))
    ("max_commodity_weight", "maximum weight of a single commodity", cxxopts::value<int>()->default_value("10"))
    ("capacity_looseness", "maximum unused capacity", cxxopts::value<int>()->default_value("100"))
    ("seed", "random seed", cxxopts::value<int>()->default_value("19"))
    ("time", "Time limit in ms", cxxopts::value<int>()->default_value("6000"))
    ("d_l", "LDS cutoff", cxxopts::value<int>()->default_value("50"))
    ("model", "model to run", cxxopts::value<string>())
    ("dominance", "enable dominance pruning", cxxopts::value<bool>()->default_value("1"))
    ("cache", "enable cache", cxxopts::value<bool>()->default_value("1"));
    auto result = options.parse(argc, argv);

    OptionsPDTSP opt("PDTSP problem",
                        result["size"].as<int>(),
                        result["m_commodities"].as<int>(),
                        result["grid_size"].as<int>(),
                        result["max_commodity_weight"].as<float>(),
                        result["capacity_looseness"].as<float>(),
                        result["cache"].as<bool>(),
                        result["temperature"].as<float>(),
                        result["dominance"].as<bool>()
                        );

    opt.model(PDTSP_DP::RL_DQN, "rl-dqn", "use DQN algorithm");
    opt.model(PDTSP_DP::RL_PPO, "rl-ppo", "use RL with PPO (implies restarts)");
    opt.model(PDTSP_DP::NN_HEURISTIC, "nearest", "use NN heuristic");
    // TODO switch case on the model?
    opt.model(stringToModel(result["model"].as<string>()));

    opt.solutions(0);
    opt.seed(result["seed"].as<int>());
    opt.time(result["time"].as<int>());
    opt.d_l(result["d_l"].as<int>());

    SimpleSearchTracer* tracer = new SimpleSearchTracer();

    if(opt.model() == PDTSP_DP::RL_DQN || opt.model() == PDTSP_DP::NN_HEURISTIC) {
        Search::Options o;
        Search::TimeStop ts(opt.time());
        o.stop = &ts;
        PDTSP_DP* p = new PDTSP_DP(opt);
        o.d_l = opt.d_l();
        o.tracer = tracer;
        LDS<PDTSP_DP> engine(p, o);
        delete p;
        cout << "nb_cities,seed,time,tour_cost,depth" << std::endl;
        while(PDTSP_DP* p = engine.next()) {
            int cur_cost = p->cost().val();
            if(cur_cost < best_cost) {
                best_cost = cur_cost;
                int depth = engine.statistics().depth;
                cout << p->to_string() << "," << depth << endl ;
            }
            delete p;
        }

        std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
        int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds> (end-start).count();
        cout << elapsed_seconds << "ms" << endl;
        cout << "max-depth reached: " << engine.statistics().depth << endl;

        if(engine.stopped()){
            cout << "TIMEOUT" << endl;
        }
        else{
            cout << "FOUND OPTIMAL" << endl;
        }
    }
    else if(opt.model() == PDTSP_DP::RL_PPO) {
        Search::Options o;

        Search::TimeStop ts(opt.time());
        o.stop = &ts;
        PDTSP_DP* p = new PDTSP_DP(opt);

        Search::Cutoff* c = Search::Cutoff::luby(result["luby"].as<int>());
        o.cutoff = c;
        o.tracer = tracer;
        RBS<PDTSP_DP,BAB> engine(p,o);
        delete p;

        cout << "nb_cities,seed,time,tour_cost,num_nodes,num_fails" << std::endl;
        while(PDTSP_DP* p = engine.next()) {
            int cur_cost = p->cost().val();
            if(cur_cost < best_cost) {
                best_cost = cur_cost;
                int num_nodes = engine.statistics().node;
                int num_fail = engine.statistics().fail;
                cout << p->to_string() << "," << num_nodes << "," << num_fail << endl ;
            }
            delete p;
        }
        std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
        int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds> (end-start).count();
        cout << elapsed_seconds << "ms" << endl;

        if(engine.stopped()){
            cout << "TIMEOUT" << endl;
        }
        else{
            cout << "FOUND OPTIMAL" << endl;
        }
    }

    else {
        cout << "Model not implemented" << std::endl;
    }

    return 0;
}



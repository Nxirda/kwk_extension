#include <iostream>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using usz = std::size_t;
using f64 = double;

/* Constants */
static constexpr usz WARMUPS = 5;
static constexpr usz METRICS = 7;
static constexpr usz LOOPING = 5;

void print_results(std::vector<std::vector<f64>> results)
{

}

//
template<typename Func, typename... Args>
auto run(Func &func, Args&&... args)
{
    using namespace std::chrono;
    
    auto start = high_resolution_clock::now();

    func(std::forward<Args>(args)...);

    auto stop  = high_resolution_clock::now();
    
    auto duration = duration_cast<milliseconds>(stop - start).count();
    return duration;
}

//
template<typename Func, typename... Args>
auto benchmark_runtime(usz iter, Func& func, Args&&... args)
{

    usz i = 0;
    /*   Warmpup   */
    for(i = 0; i < WARMUPS; ++i)
    {
        run(func, std::forward<Args>(args)...);
    }
   
    /* Actual Benchmark */
    std::vector<f64> stats(iter, 0);
    for(i = 0; i < iter; ++i)
    {
        stats[i] = run(func, std::forward<Args>(args)...);
    }
    
    std::vector<f64> result(8, 0);
    std::sort(stats.begin(), stats.end());

    /*   Metrics   */
    auto acc = std::accumulate(stats.begin(), stats.end(), 0);
    double mean = acc * (1/stats.size());
    
    double median = 0;
    if(stats.size() % 2 == 0)
    {
        median = (stats[stats.size()/2 -1] + stats[stats.size()/2])/2.0;
    }
    else
    {
        median = stats[stats.size()/2];
    }

    double std_dev = 0;
    for(const auto &time : stats)
    {
        std_dev += std::pow((time - mean), 2);
    }
    std_dev = std::sqrt(std_dev/stats.size());

    double min = stats.front();
    double max = stats.back();
    
    //auto [per25, per75] = compute_percentiles(stats);

    double per25 = stats[stats.size()/4];
    double per75 = stats[(stats.size() * 3)/4];
    
    std::vector<f64> res(METRICS);
    res[0] = mean;
    res[1] = median;
    res[2] = std_dev;
    res[3] = min;
    res[4] = max;
    res[5] = per25;
    res[6] = per75;
    return res;
}

//


//
void foo(void)
{
    using namespace std::chrono_literals;
    for(int i = 0; i < 100; ++i)
    {
        std::this_thread::sleep_for(10ms);
    }
}

int main()
{
    benchmark_runtime(5,foo);
    return 0;
}

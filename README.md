# Reliability Optimization

This project addresses an optimization problem aiming to increase network reliability.

## Getting Started

### Prerequisites

To run this application it is necessary to have [Gurobi](https://www.gurobi.com/) problem solver installed. Also, this project uses python3 and some almost default libraries, as we list below.

* argparse
* json
* operator
* sys
* collections
* uuid
* matplotlib
* Numpy
* Scipy
* Simpy
* Shapely

```sudo -H pip3 install numpy scipy matplotlib simpy shapely```

### Usage

There are three possible uses for the code in this repo. First, it can simulate a vehicular wireless network with the standard handover (baseline), with heuristic handover and an optimization problem. Second, the code can generate and plot instances with homogeneous or clustered obstacles placement. Third, it is possible to import the individual modules in the directory fivegmodules. 

So, to use the first functionality you can use:

```./main.py --ttt <your_ttt> -i <your_instance> [-p] [--untoggle-blockage]```

which will run the baseline handover with 640 milliseconds TTT, by default. To generate your own instaces:

```./instance-loader.py --vx <V_X> --vy <V_Y> -b <blockage_density [b/km2]> -s <seed> --clustered [rate, sigma]```

And finally

You can also run a bunch of simulations with only one command:

```./simulation.py ```


## License

This project is licensed under GNU GPL v3 [License](LICENSE.md). If you use any data or code contained in this repository, please cite and 
notify the authors.

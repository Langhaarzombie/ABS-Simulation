# Bachelor Thesis concerning Active Brownian Spheres undergoing Phase Transitions

## How to use

After installing all the required packages listed in `requirements.txt` you can run the program.
There are two options for you:

- Generate data and show simulation
- Take existing data and show the simulation

Generating data and then showing the simulation is the default behaviour.
Therefore, you can run the program simply with `python main.py`.

If you just want to show some already generated data, you need to append *-s* or *--show*.
Consequently, your command will look like `python main.py --show`.

## Configuration

Located in `main.py` there is a config variable that contains necessary configurations to run the program.

- `save_file`: filename where simulation data is written to and read from
- `window_size`: dimensions of the cube the spheres move in
- `sphere_count`: number of spheres to simulate
- `simulation_timestep`: size of one timestep in the simulation
- `simulation_steps`: how many timesteps are calculated
- `temperature`: init value to define velocities of spheres
- `sigma`: used in LJ potential, also defines interaction range

*NOTE that all the configurations are for running and showing the simulation.
Showing data that was generated with a different configuration might result in unexpected behaviour.*

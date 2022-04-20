# Bachelor Thesis concerning Active Brownian Spheres undergoing Phase Transitions

## How to use

After installing all the required packages listed in `requirements.txt` you can run the program.
There are two options for you:

- Generate data and show simulation
- Take existing data and show the simulation

Generating data and then showing the simulation is the default behaviour.
Therefore, you can run the program simply with `python main.py`.

For detailled information on how to only show data or use specific inits, use `python main.py --help`.

## Configuration

For configuring the simulation the file `abs_config.yml` (by default) is used.
In this file, the following parameters need to be set:

- `count`: number of spheres to simulate (default: 20)
- `bounds`: bounds of simulated cube (default: 2)
- `timestep`: size of timestep (default: 0.0005)
- `steps`: number to timesteps to simulate (default: 100)
- `sigma`: used in WCA potential & defines interaction range (default: 1)
- `temperature`: temperature of initialization (default: 200)
- `init`:
    - `save_file`: filename for storing init config in (default: `abs_init.csv`)
- `run`:
    - `save_file`: filename for storing simulation data (default: `abs_simulation.csv`)
    - `observables`: collection of observables and how frequently they are saved in units of timesteps (default: 1 for all of the following)
        - `position`
        - `velocity`
        - `potential_energy`
        - `kinetic_energy`

NOTE that in order to show simulation data correctly, the `count` and `steps` value in the config file must be the same when generating and showing the data.

For now it is not possible to omit certain parameters.
Thus it is advised to copy the default config to `abs_config.yml` and start from there.


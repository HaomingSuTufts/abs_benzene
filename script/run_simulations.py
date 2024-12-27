#run FEP simulation for molecule
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import pickle
import os
import time
import argparse
import logging
import functools
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from utils import setup_logging, monitor_performance
from config import SimulationConfig

class PathManager:

    def __init__(self, phase: str):
        self.base_dir = Path(f"./output/{phase}_phase")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def lambdas_file(self) -> Path:
        return self.base_dir / "lambdas.pkl"
    
    @property
    def topology_file(self) -> Path:
        return self.base_dir / "topology.pkl"
    
    @property
    def pdb_file(self) -> Path:
        return self.base_dir / "system.pdb"
    
    @property
    def save_dir(self) -> Path:
        return self.base_dir / "sys"


def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(description="Run FEP simulation for molecule")
    parser.add_argument("--phase", type= str, help="Phase of the molecule",default="water")
    return parser.parse_args()


def get_slurm_array_task_id():
    """Get SLURM array task ID."""
    return int(os.environ["SLURM_ARRAY_TASK_ID"])


def load_lambdas(phase):
    """Load lambda values from file."""
    with open(f"./output/{phase}_phase/lambdas.pkl", "rb") as f:
        return pickle.load(f)
    
def load_system(phase, lambdas):
    """Load serialized system from file."""
    elec, vdw = lambdas[0][0], lambdas[0][1]
    with open(f"./output/{phase}_phase/sys/{elec:.2f}_{vdw:.2f}.xml", "r") as f:
        return mm.XmlSerializer.deserialize(f.read())
    
def add_barostat(system, phase):
    """Add barostat to system."""
    if phase == "water":
        system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 298.15 * unit.kelvin))
    return system

def load_topology(phase):
    """Load topology from file."""
    with open(f"./output/{phase}_phase/topology.pkl", "rb") as f:
        return pickle.load(f)
    
def load_pdb(phase):
    """Load PDB file."""
    return app.PDBFile(f"./output/{phase}_phase/system.pdb")

def create_simulation(topology, system, pdb):
    """Create OpenMM simulation object."""
    system_setting = {
        "temperature": SimulationConfig.DEFAULT_SETTINGS["temperature"],
        "friction": SimulationConfig.DEFAULT_SETTINGS["friction"],
        "timestep": SimulationConfig.DEFAULT_SETTINGS["timestep"],
    }
    integrator = mm.LangevinMiddleIntegrator(system_setting["temperature"], system_setting["friction"], system_setting["timestep"])
    platform = mm.Platform.getPlatformByName("CUDA")if os.getenv("CUDA_VISIBLE_DEVICES") else mm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    return simulation

def minimize_energy(simulation):
    """Minimize energy of system."""
    simulation.minimizeEnergy(tolerance=1.0)
    return simulation

def equilibrate(simulation, temperature):
    """Equilibrate system at given temperature."""
    simulation.integrator.setStepSize(0.001 * unit.picoseconds)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
    logging.info(f"Equilibrating at {temperature} K")
    simulation.step(10_000)
    return simulation

@monitor_performance
def run_simulation(simulation: app.Simulation, 
                  config: SimulationConfig,
                  lambda_str: str,
                  phase: str, *args, **kwargs) -> app.Simulation:

    simulation.integrator.setStepSize(config.DEFAULT_SETTINGS['timestep'] )
    os.makedirs(f"./output/{phase}_phase/traj", exist_ok=True)

    simulation.reporters.append(app.DCDReporter(f"./output/{phase}_phase/traj/{lambda_str}.dcd", config.DEFAULT_SETTINGS['trajectory_interval']))
    logging.info("Running simulation")
    simulation.step(config.DEFAULT_SETTINGS['simulation_steps'])
    simulation.saveCheckpoint(f"./output/{phase}_phase/traj/{lambda_str}_checkpoint.chk")
    return simulation    

def main():
    try:
        args = get_args()
        paths = PathManager(args.phase)
        log_path = paths.base_dir
        setup_logging(log_path, task_name='run_simulation')
        config = SimulationConfig()
        
        logging.info(f"Starting simulation for phase: {args.phase}")
        
        if os.getenv("SLURM_ARRAY_TASK_ID") is not None:
            idx_lambda = get_slurm_array_task_id()
        else:
            idx_lambda = 0
        lambdas_list = load_lambdas(args.phase)
        lambdas = lambdas_list[idx_lambda]
        elec, vdw = lambdas[0][0], lambdas[0][1]
        lambdas_str = f"{elec:.2f}_{vdw:.2f}"
        logging.info(f"Running simulation for lambdas {lambdas_str}")

        system = load_system(args.phase, lambdas)
        system = add_barostat(system, args.phase)
        topology = load_topology(args.phase)
        pdb = load_pdb(args.phase)
        simulation = create_simulation(topology, system, pdb)
        simulation = minimize_energy(simulation)
        
        temperatures = np.linspace(config.DEFAULT_SETTINGS['heat_bath_start']._value()
                                    ,config.DEFAULT_SETTINGS['temperature']._value()
                                    ,config.DEFAULT_SETTINGS['heat_bath_step_size'].value())
        for T in temperatures:
            simulation = equilibrate(simulation, T)
        
        simulation = run_simulation(simulation,  config, lambdas_str, args.phase)
        
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

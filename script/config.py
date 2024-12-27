# config file
from hmac import new
from jax import config
import openmm.app as app
from simtk import unit
import yaml
import os


def tuple_representer(dumper, data):
    return dumper.represent_list(list(data))


def unit_representer(dumper, data):
    return dumper.represent_scalar('!unit', str(data.unit))


yaml.add_representer(tuple, tuple_representer)
yaml.add_representer(unit.Quantity, unit_representer)


class SimulationConfig:

    DEFAULT_SETTINGS = {
        'temperature': 298.15 * unit.kelvin,
        'pressure': 1.0 * unit.atmosphere,
        'friction': 1.0 / unit.picosecond,
        'timestep': 0.002 * unit.picoseconds,
        'equilibration_steps': 10_000,
        'simulation_steps': 5_000_000,
        'nonbondedMethod': app.PME,
        'nonbondedCutoff': 1.0 * unit.nanometer,
        'constraints': app.HBonds,
        'switchDistance': 0.9 * unit.nanometer,
        'lenth_unit': unit.nanometer,
        'total_simulation_steps': 5_000_000,
        'equilibration_steps': 10_000,
        'trajectory_interval': 1000,
        'kbT': 298.15 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA,
        'heat_bath_start': 50 * unit.kelvin,
        'heat_bath_step_size': 10 * unit.kelvin,
        'report_energy_unit': unit.kilocalorie_per_mole,
    }

    lambda_list = [
        [(1.0, 1.0)],
        [(0.8, 1.0)],
        [(0.6, 1.0)],
        [(0.4, 1.0)],
        [(0.2, 1.0)],
        [(0.0, 1.0)],
        [(0.0, 0.95)],
        [(0.0, 0.9)],
        [(0.0, 0.8)],
        [(0.0, 0.7)],
        [(0.0, 0.6)],
        [(0.0, 0.5)],
        [(0.0, 0.4)],
        [(0.0, 0.3)],
        [(0.0, 0.2)],
        [(0.0, 0.1)],
        [(0.0, 0.05)],
        [(0.0, 0.0)],

    ]

    def __init__(self, config_file=None):
        self.settings = self.DEFAULT_SETTINGS.copy()
        if config_file:
            self.load(config_file)

    def get_settings(self):
        return self.settings

    def get_lambda_list(self):
        return self.lambda_list

    def _convert_units_to_str(self, settings):
        converted = {}
        for key, value in settings.items():
            if isinstance(value, unit.Quantity):
                converted[key] = {
                    'value': value._value,
                    'unit': str(value.unit)
                }
            else:
                converted[key] = value
        return converted

    def _convert_str_to_units(self, settings):

        converted = {}
        for key, value in settings.items():
            if isinstance(value, dict) and 'unit' in value:

                unit_str = value['unit']
                if hasattr(unit, unit_str):
                    converted[key] = value['value'] * getattr(unit, unit_str)
                else:
                    converted[key] = value
            else:
                converted[key] = value
        return converted

    def save(self, config_file):
        with open(config_file, 'w') as f:
            converted_settings = {}
            for key, value in self.settings.items():
                if isinstance(value, unit.Unit):
                    converted_settings[key] = str(value)
                elif isinstance(value, unit.Quantity):
                    converted_settings[key] = {
                        'value': float(value._value),
                        'unit': str(value.unit)
                    }
                elif value == app.PME:
                    converted_settings[key] = 'PME'
                elif value == app.HBonds:
                    converted_settings[key] = 'HBonds'
                else:
                    converted_settings[key] = value

            lambda_data = [
                {'charge': item[0][0], 'vdw': item[0][1]}
                for item in self.lambda_list
            ]
            converted_settings['lambda_list'] = lambda_data

            yaml.dump(converted_settings, f, default_flow_style=False)

    def load(self, config_file):
        with open(config_file, 'r') as f:
            loaded_data = yaml.safe_load(f)

            for key, value in loaded_data.items():
                if key == 'lambda_list':
                    continue
                if isinstance(value, dict) and 'unit' in value:
                    unit_str = value['unit']
                    if hasattr(unit, unit_str):
                        self.settings[key] = float(
                            value['value']) * getattr(unit, unit_str)
                elif isinstance(value, str) and hasattr(unit, value):
                    self.settings[key] = getattr(unit, value)
                elif value == 'PME':
                    self.settings[key] = app.PME
                elif value == 'HBonds':
                    self.settings[key] = app.HBonds
                else:
                    self.settings[key] = value

            if 'lambda_list' in loaded_data:
                self.lambda_list = [
                    [(item['charge'], item['vdw'])]
                    for item in loaded_data['lambda_list']
                ]

            # check for report_energy_unit
            if 'report_energy_unit' in self.settings:
                if isinstance(self.settings['report_energy_unit'], str):
                    if self.settings['report_energy_unit'] == 'kilocalorie/mole':
                        self.settings['report_energy_unit'] = unit.kilocalorie_per_mole
                    elif self.settings['report_energy_unit'] == 'kilojoule/mole':
                        self.settings['report_energy_unit'] = unit.kilojoule_per_mole
                    else:
                        raise ValueError(f"Unknown unit: {
                                         self.settings['report_energy_unit']}")


if __name__ == '__main__':
    ...

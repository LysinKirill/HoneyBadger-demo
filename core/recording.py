import json
import numpy as np
from typing import List, Dict, Any
from datetime import datetime


class OptimizationRecording:
    def __init__(self):
        self.function_name = ""
        self.function_dim = 0
        self.bounds = None
        self.selected_vars = []
        self.fixed_values = []
        self.population_history = []
        self.best_solution_history = []
        self.fitness_history = []
        self.timestamp = ""

    def record_step(self, population: np.ndarray, best_solution: np.ndarray, best_fitness: float):
        self.population_history.append(population.copy())
        self.best_solution_history.append(best_solution.copy())
        self.fitness_history.append(best_fitness)

    def save(self, filename: str) -> str:
        data = {
            'function_name': self.function_name,
            'function_dim': self.function_dim,
            'bounds': self.bounds,
            'selected_vars': self.selected_vars,
            'fixed_values': self.fixed_values,
            'timestamp': self.timestamp or datetime.now().isoformat(),
            'population_history': [pop.tolist() for pop in self.population_history],
            'best_solution_history': [sol.tolist() for sol in self.best_solution_history],
            'fitness_history': self.fitness_history
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        return filename

    def load(self, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)

        self.function_name = data['function_name']
        self.function_dim = data['function_dim']
        self.bounds = data['bounds']
        self.selected_vars = data['selected_vars']
        self.fixed_values = data['fixed_values']
        self.timestamp = data['timestamp']
        self.population_history = [np.array(pop) for pop in data['population_history']]
        self.best_solution_history = [np.array(sol) for sol in data['best_solution_history']]
        self.fitness_history = data['fitness_history']

        return self

    def get_step(self, index: int) -> Dict[str, Any]:
        if 0 <= index < len(self.population_history):
            return {
                'population': self.population_history[index],
                'best_solution': self.best_solution_history[index],
                'best_fitness': self.fitness_history[index],
                'step': index,
                'total_steps': len(self.population_history)
            }
        return None

    @property
    def total_steps(self) -> int:
        return len(self.population_history)
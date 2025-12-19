import numpy as np
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass
import numpy.typing as npt


@dataclass
class HBAParams:
    pop_size: int = 30
    max_iter: int = 500
    C: float = 2.0
    beta: float = 6.0
    seed: Optional[int] = None


class HoneyBadgerAlgorithm:
    def __init__(self, params: HBAParams = HBAParams()):
        self.params = params
        self.rng = np.random.default_rng(params.seed)

        self.population: npt.NDArray[np.float64] | None = None
        self.fitness: npt.NDArray[np.float64] | None = None
        self.best_solution: npt.NDArray[np.float64] | None = None
        self.best_fitness: float = float('inf')
        self.convergence_curve: List[float] = []
        self.current_iter: int = 0

    def initialize_population(self, dim: int, bounds: Tuple[float, float]) -> None:
        lower, upper = bounds
        self.population = self.rng.uniform(
            lower, upper, (self.params.pop_size, dim)
        )

    def evaluate(self, func: Callable[[npt.NDArray], float]) -> None:
        self.fitness = np.array([func(ind) for ind in self.population])

        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()

    def compute_intensity(self) -> npt.NDArray:
        r_idx1, r_idx2 = self.rng.choice(self.params.pop_size, 2, replace=False)
        S = np.sum((self.population[r_idx1] - self.population[r_idx2]) ** 2)

        distances = np.linalg.norm(self.best_solution - self.population, axis=1)
        distances = np.where(distances == 0, 1e-12, distances)

        r5 = self.rng.random()

        intensity = r5 * S / (4 * np.pi * distances ** 2)
        return intensity

    def update_density_factor(self) -> float:
        return self.params.C * np.exp(-self.current_iter / self.params.max_iter)

    def get_flag_direction(self) -> int:
        return 1 if self.rng.random() < 0.5 else -1

    def update_position_digging(self, alpha: float, intensity: npt.NDArray) -> npt.NDArray:
        new_population = np.zeros_like(self.population)

        for i in range(self.params.pop_size):
            r2, r3, r4 = self.rng.random(3)
            F = self.get_flag_direction()

            d_i = self.best_solution - self.population[i]

            new_population[i] = (
                    self.best_solution
                    + F * self.params.beta * intensity[i] * self.best_solution
                    + F * r2 * alpha * d_i
                    * np.abs(np.cos(2 * np.pi * r3) * (1 - np.cos(2 * np.pi * r4)))
            )

        return new_population

    def update_position_honey(self, alpha: float) -> npt.NDArray:
        new_population = np.zeros_like(self.population)

        for i in range(self.params.pop_size):
            r7 = self.rng.random()
            F = self.get_flag_direction()
            d_i = self.best_solution - self.population[i]
            new_population[i] = self.best_solution + F * r7 * alpha * d_i

        return new_population

    def optimize(self,
                 func: Callable[[npt.NDArray], float],
                 dim: int,
                 bounds: Tuple[float, float]) -> Tuple[npt.NDArray, float]:
        self.initialize_population(dim, bounds)
        self.evaluate(func)
        self.convergence_curve = [self.best_fitness]
        self.current_iter = 0

        for iter in range(self.params.max_iter):
            alpha = self.update_density_factor()
            intensity = self.compute_intensity()

            if self.rng.random() < 0.5:
                new_population = self.update_position_digging(alpha, intensity)
            else:
                new_population = self.update_position_honey(alpha)

            lower, upper = bounds
            new_population = np.clip(new_population, lower, upper)

            old_population = self.population.copy()
            self.population = new_population
            self.evaluate(func)

            for i in range(self.params.pop_size):
                old_fitness = func(old_population[i])
                if old_fitness < self.fitness[i]:
                    self.population[i] = old_population[i]
                    self.fitness[i] = old_fitness

            self.convergence_curve.append(self.best_fitness)
            self.current_iter += 1

        return self.best_solution, self.best_fitness

    def run_one_iteration(self) -> None:
        if self.current_iter >= self.params.max_iter:
            return

        self.previous_population = self.population.copy()
        alpha = self.update_density_factor()
        intensity = self.compute_intensity()
        self.last_intensity = intensity
        self.current_phase = "Digging" if np.random.random() < 0.5 else "Honey"

        if np.random.random() < 0.5:
            new_population = self.update_position_digging(alpha, intensity)
        else:
            new_population = self.update_position_honey(alpha)

        lower, upper = self.current_bounds
        new_population = np.clip(new_population, lower, upper)

        old_population = self.population.copy()
        self.population = new_population
        self.evaluate(self.current_func)

        for i in range(self.params.pop_size):
            old_fitness = self.current_func(old_population[i])
            if old_fitness < self.fitness[i]:
                self.population[i] = old_population[i]
                self.fitness[i] = old_fitness

        self.convergence_curve.append(self.best_fitness)
        self.current_iter += 1

    def set_optimization_problem(self, func, dim, bounds):
        self.current_func = func
        self.current_dim = dim
        self.current_bounds = bounds
        self.initialize_population(dim, bounds)
        self.evaluate(func)

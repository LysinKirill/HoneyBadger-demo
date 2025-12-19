import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt


class DataExporter:

    @staticmethod
    def get_io_dir() -> Path:
        io_dir = Path("iodata/exports")
        io_dir.mkdir(parents=True, exist_ok=True)
        return io_dir

    @staticmethod
    def export_to_json(data: Dict, filename: str = None, subfolder: str = None) -> str:
        export_dir = DataExporter.get_io_dir()

        if subfolder:
            export_dir = export_dir / subfolder
            export_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hba_results_{timestamp}.json"

        if not filename.endswith('.json'):
            filename += '.json'

        filepath = export_dir / filename

        json_data = DataExporter._prepare_for_json(data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    @staticmethod
    def export_convergence_csv(convergence_data: List[float],
                               filename: str = None,
                               subfolder: str = None) -> str:
        export_dir = DataExporter.get_io_dir()

        if subfolder:
            export_dir = export_dir / subfolder
            export_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"convergence_{timestamp}.csv"

        if not filename.endswith('.csv'):
            filename += '.csv'

        filepath = export_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'best_fitness'])
            for i, fitness in enumerate(convergence_data):
                writer.writerow([i, fitness])

        return str(filepath)

    @staticmethod
    def export_solutions_csv(solutions_history: List[npt.NDArray],
                             filename: str = None,
                             subfolder: str = None) -> Optional[str]:
        if not solutions_history:
            return None

        export_dir = DataExporter.get_io_dir()

        if subfolder:
            export_dir = export_dir / subfolder
            export_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solutions_history_{timestamp}.csv"

        if not filename.endswith('.csv'):
            filename += '.csv'

        filepath = export_dir / filename

        dim = len(solutions_history[0])

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ['iteration'] + [f'x{i}' for i in range(dim)]
            writer.writerow(headers)

            for i, solution in enumerate(solutions_history):
                row = [i] + solution.tolist()
                writer.writerow(row)

        return str(filepath)

    @staticmethod
    def export_engineering_results(problem_name: str,
                                   solution: npt.NDArray,
                                   objective_value: float,
                                   constraints: List[Dict],
                                   filename: str = None) -> str:
        export_dir = DataExporter.get_io_dir() / "engineering"
        export_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = problem_name.replace(' ', '_').replace('/', '_')
            filename = f"{safe_name}_results_{timestamp}.json"

        if not filename.endswith('.json'):
            filename += '.json'

        filepath = export_dir / filename

        data = {
            'problem': problem_name,
            'timestamp': datetime.now().isoformat(),
            'solution': solution.tolist(),
            'objective_value': objective_value,
            'constraints': constraints
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    @staticmethod
    def export_algorithm_params(params: Dict, filename: str = None) -> str:
        config_dir = Path("iodata/configs")
        config_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hba_params_{timestamp}.json"

        if not filename.endswith('.json'):
            filename += '.json'

        filepath = config_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        return str(filepath)

    @staticmethod
    def get_recent_exports(limit: int = 10, subfolder: str = None) -> List[Path]:
        export_dir = DataExporter.get_io_dir()

        if subfolder:
            export_dir = export_dir / subfolder

        if not export_dir.exists():
            return []

        files = list(export_dir.glob("*"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[:limit]

    @staticmethod
    def _prepare_for_json(data: Dict) -> Dict:
        if isinstance(data, dict):
            return {k: DataExporter._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [DataExporter._prepare_for_json(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.generic):
            return data.item()
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        else:
            return data
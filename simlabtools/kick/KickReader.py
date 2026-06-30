import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


class KickReader:

    def __init__(self, filepath: str | Path):
        """
        Initialise a KickReader object.

        The project directory must have the following structure:

            {project_folder}/
            ├── {name}.txt
            └── Kamera/
                └── {name}.cine

        The corresponding `.cine` file is assumed to have the same
        base name as the text file and to be located in the `Kamera`
        subdirectory.

        Parameters
        ----------
        filepath : str or Path
            Path to the KICK text file (`{name}.txt`).
        """

        if isinstance(filepath, str):
            self.filepath = Path(filepath)
        else:
            self.filepath = filepath

        self.name = self.filepath.stem

        self.meta = {}
        self.kick = {}
        self.camera = {}

        self.camera_idx: int = 0
        self.kick_idx: int = 0


        self.read()

    def read(self):
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.meta, self.kick = self._parse_kick(lines)

        camera_file = (
            self.filepath.parent /
            'Kamera' /
            f'{self.filepath.stem}.pps'
        )

        if not camera_file.exists():
            raise FileNotFoundError(camera_file, 'not found')

        with open(camera_file, 'r') as f:
            lines = f.readlines()
        meta_temp, self.camera = self._parse_camera(lines)
        self.meta |= meta_temp

    def _parse_kick(self, lines):

        meta = {}
        data_start = 0

        for i, line in enumerate(lines):

            line = line.strip()

            if line.startswith("Time\t"):
                data_start = i + 2
                break

            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()

        rows = []

        for line in lines[data_start:]:

            line = line.strip()

            if not line:
                continue

            vals = [
                float(v.replace(",", "."))
                for v in line.split("\t")
            ]

            rows.append(vals)

        arr = np.asarray(rows)

        data = {
            "time": arr[:, 0],
            "photo_cell": arr[:, 1],
            "laser_trolley": arr[:, 2],
            "laser_wall": arr[:, 3],
            "load_cell_1": arr[:, 4],
            "load_cell_2": arr[:, 5],
        }

        return meta, data

    def _parse_camera(self, lines):

        first = [x.strip() for x in lines[0].split(",")]

        meta = {
            "type": first[0],
            "id": int(first[1]),
            "cine_file": first[2],
            "unit": first[3],
            "scale": float(first[6])
        }

        n = len(lines) - 2

        image_nr = np.empty(n, dtype=int)
        time_from_trig = np.empty(n)
        x = np.empty(n)
        y = np.empty(n)

        abs_time = []

        for i, line in enumerate(lines[2:]):

            parts = line.strip().split(",", 4)

            image_nr[i] = int(parts[0])
            time_from_trig[i] = float(parts[1])
            x[i] = float(parts[2])
            y[i] = float(parts[3])

            tstr = parts[4].rstrip(",")

            date_part, frac = tstr.rsplit(".", 1)

            dt = datetime.strptime(
                date_part,
                "%a %b %d %Y %H:%M:%S.%f"
            )

            dt += timedelta(
                microseconds=int(frac)/1000
            )

            abs_time.append(dt)

        abs_time = np.array(abs_time)

        t0 = abs_time[0]

        abs_time_sec = np.array([
            (t - t0).total_seconds()
            for t in abs_time
        ])

        data = {
            "image_nr": image_nr,
            "time_from_trig": time_from_trig,
            "x": x,
            "y": y,
            "abs_time": abs_time,
            "abs_time_sec": abs_time_sec
        }

        return meta, data

    def sync(
            self,
            kick_idx: int | None = None,
            camera_idx: int | None = None
    ):

        if kick_idx is not None:
            self.kick_idx = kick_idx
        if camera_idx is not None:
            self.camera_idx = camera_idx

        dt_force = 0.02

        self.kick["time_sync"] = (
            np.arange(len(self.kick["time"]))
            - self.kick_idx
        ) * dt_force

        self.camera["time_sync"] = (
            self.camera["time_from_trig"]
            - self.camera["time_from_trig"][self.camera_idx]
        ) * 1000

        self.interpolate_force()

    def interpolate_force(self):

        f = PchipInterpolator(
            self.kick["time_sync"],
            self.kick["load_cell_2"],
            extrapolate=True
        )

        self.camera["force"] = f(
            self.camera["time_sync"]
        )

        # self.force = self.camera['force'][self.camera_idx:]
        # self.displacement = (
        #     (self.camera['x']**2 + self.camera['y']**2)**(0.5)
        # )[self.camera_idx:]
        # self.displacement -= self.displacement[0]
        # self.time = self.camera['time_sync'][self.camera_idx:] - self.camera['time_sync'][self.camera_idx]

    def plot_force(self):
        ...

    @property
    def force(self):

        if 'force' not in self.camera:
            raise RuntimeError('Call sync() first.')
        
        return self.camera['force'][self.camera_idx:]
    
    @property
    def displacement(self):

        disp = np.sqrt(
            self.camera['x']**2 + 
            self.camera['y']**2
        )

        disp = disp[self.camera_idx:]

        return disp - disp[0]
    
    @property
    def time(self):

        t = self.camera['time_sync'][self.camera_idx:]

        return t - t[0]
    
    @property
    def _velocity(self):

        disp = np.sqrt(
            self.camera['x']**2 + 
            self.camera['y']**2
        )

        disp_smooth = savgol_filter(
            disp,
            window_length=50,
            polyorder=3
        )

        return np.gradient(
            disp,
            self.camera['time_from_trig']
        )
    
    @property
    def velocity(self):

        if not hasattr(self, 'camera_idx'):
            raise RuntimeError('Call sync() first')
        
        return self._velocity[self.camera_idx:]
    
    @property
    def initial_velocity(self):

        if not hasattr(self, 'camera_idx'):
            raise RuntimeError('Call sync() first')

        return np.mean(
            self._velocity[:self.camera_idx]
        )
    
    def summary(self) -> dict:
        """
        Returns a summary of the experiment

        Returns
        -------
        dict    
            Dictionary containing key experimental metrics.
        """

        if self.time is None:
            raise RuntimeError(
                "Call sync() before requesting summary"
            )
        
        return {
            "name": self.name,

            "initial_velocity_m_s":
                self.initial_velocity / 1000,
            
            "max_force_kN":
                np.max(self.force),

            "max_displacement_mm":
                np.max(self.displacement),
            
            "test_duration_ms":
                np.max(self.time),
            
            "n_force_samples":
                len(self.kick["time"]),
            
            "n_camera_frames":
                len(self.camera["image_nr"]),

            "impact_force_idx":
                self.kick_idx,

            "impact_camera_idx":
                self.camera_idx

        }
    
    def print_summary(self):
        """
        Alternative to ``pprint(self.summary())``
        """

        s = self.summary()

        print(f"\n{'='*50}")
        print(f"Experiment summary: {s['name']}")
        print(f"{'='*50}")

        print(
            f"Initial velocity      : "
            f"{s['initial_velocity_m_s']:.2f} m/s"
        )

        print(
            f"Maximum force         : "
            f"{s['max_force_kN']:.2f} kN"
        )

        print(
            f"Maximum displacement  : "
            f"{s['max_displacement_mm']:.2f} mm"
        )

        print(
            f"Test duration         : "
            f"{s['test_duration_ms']:.2f} ms"
        )

        print(
            f"Camera frames         : "
            f"{s['n_camera_frames']}"
        )

        print(
            f"Force samples         : "
            f"{s['n_force_samples']}"
        )
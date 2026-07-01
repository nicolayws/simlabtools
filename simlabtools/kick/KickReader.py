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
                └── {name}.pps

        The corresponding `.pps` file is assumed to have the same
        base name as the text file and to be located in the `Kamera`
        subdirectory.

        Parameters
        ----------
        filepath : str or Path
            Path to the KICK text file (`{name}.txt`)

        Raises
        ------
        FileNotFoundError
            If the specified text file does not exist or if the
            corresponding `.pps` file cannot be found in the
            `Kamera` subdirectory.
        ValueError
            If the text file does not contain valid KICK data.

        Notes
        -----
        The reader automatically locates and reads both the KICK text
        file and the associated camera metadata file (`.pps`). The
        loaded data are stored internally and made available through
        the class properties and methods.

        Examples
        --------
        Load a KICK test:

        >>> from simlabtools.kick import KickReader
        >>> test = KickReader("D:/Tests/Test01/Test01.txt")

        Sync the data from kicking machine and high-speed camera

        >>> test.sync(1017, 617)

        Access the recorded force data:

        >>> force = test.force
        >>> time = test.time

        Access camera information from the corresponding `.pps` file:

        >>> fps = test.camera_fps
        >>> print(f"Camera frame rate: {fps} fps")
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

        self._smoothing = None

        self.read()

    def read(self):
        """
        Read data
        """
        
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
        """
        Syncronising the data from the high-speed camera and 
        the kicking machine. 

        Parameters
        ----------
        kick_idx: int, optional
            Index for syncing kicking machine data, can also be set prior
            to using this method.
        camera_idx: int, optional
            Frame number from high-speed camera for syncing data, can also
            be set prior to using this method.
        """

        if kick_idx is not None:
            self.kick_idx = kick_idx
        if camera_idx is not None:
            frame = np.where(self.camera['image_nr'] > camera_idx)[0]

            if len(frame) == 0:
                raise ValueError(
                    "camera_idx must be below ", np.max(self.camera['image_nr'])
                )
            
            self.camera_idx = frame[0]

        dt_force = 0.02

        self.kick["time_sync"] = (
            np.arange(len(self.kick["time"]))
            - self.kick_idx
        ) * dt_force

        self.camera["time_sync"] = (
            self.camera["time_from_trig"]
            - self.camera["time_from_trig"][self.camera_idx]
        ) * 1000

        self._interpolate_force()

    def _interpolate_force(self):

        f = PchipInterpolator(
            self.kick["time_sync"],
            self.kick["load_cell_2"],
            extrapolate=True
        )

        self.camera["force"] = f(
            self.camera["time_sync"]
        )

    @property
    def force(self):
        """
        Corrected force array for impact. For corrected force
        with specific masses of trolley etc., call ``self.force_corrected()``
        """

        return self.force_corrected()


    @property
    def _force(self):
        """
        Uncorrected force array for impact
        """

        if 'force' not in self.camera:
            raise RuntimeError('Call sync() first.')
        
        max_idx = self.camera_idx + self.displacement.size

        force = self.camera['force'][self.camera_idx:max_idx]

        if callable(self.smoothing):

            force = self.smoothing(force)
        
        return np.asarray(force)
    
    @property
    def _displacement(self):
        """
        Displacement for entire test, not just impact
        """

        x = self.camera['x']
        y = self.camera['y']

        x -= x[0]  # Initialize
        y -= y[0]

        return np.sqrt(
            x**2 + 
            y**2
        )
    
    @property
    def displacement(self):
        """
        Displacement array for impact
        """

        disp = self._displacement

        max_idx = np.argmax(disp)

        disp = disp[self.camera_idx:max_idx]

        return disp - disp[0]
    
    @property
    def time(self):
        """
        Time array for impact
        """
        
        max_idx = self.camera_idx + self.displacement.size

        t = self.camera['time_sync'][self.camera_idx:max_idx]

        return t - t[0]
    
    @property
    def _velocity(self):

        disp = self._displacement

        disp_smooth = savgol_filter(
            disp,
            window_length=100,
            polyorder=3
        )

        return np.gradient(
            disp_smooth,
            self.camera['time_from_trig']
        )
    
    @property
    def velocity(self):
        """
        Velocity curve of the trolley during impact, computed
        as the gradient of the displacement
        """

        if not hasattr(self, 'camera_idx'):
            raise RuntimeError('Call sync() first')
        
        max_idx = self.camera_idx + self.displacement.size
        
        return self._velocity[self.camera_idx:max_idx]
    
    @property
    def initial_velocity(self):
        """
        Initial velocity of the trolley.

        Mean of ``self.velocity`` up to ``self.camera_idx``
        """

        if not hasattr(self, 'camera_idx'):
            raise RuntimeError('Call sync() first')

        return np.mean(
            self._velocity[:self.camera_idx]
        )
    
    @property
    def initial_velocity_two_point(self):
        """
        Initial velocity of the trolley, computed
        with time and position between two points;
            [1] - First visible point marker
            [2] - Time of impact
        """

        if not hasattr(self, "camera_idx"):
            raise RuntimeError("Call sync() first")
        
        dx = (
            self._displacement[self.camera_idx] - 
            self._displacement[0]
        )

        dt = (
            self.camera['abs_time_sec'][self.camera_idx] - 
            self.camera['abs_time_sec'][0]
        )

        print(f"Sampling initial velocity between frames ", end="")
        print(self.camera['image_nr'][0], self.camera['image_nr'][self.camera_idx])

        print(f"{dx = }, {dt = }")

        return dx / dt

    
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

    def force_corrected(
        self,
        mass_load_cell: float | None = None,
        mass_load_cell_after_gauge: float | None = None,
        mass_nose: float | None = None,
        mass_trolley: float | None = None
    ) -> np.ndarray:
        """
        Correct force for acceleration. If masses are not given, 
        the masses stored in the meta-data are used.

        Parameters
        mass_load_cell: float, optional
            Mass of the load cell [kg].
        mass_load_cell_after_gauge : float, optional
            Mass between the strain gauge and the load cell [kg].
        mass_nose : float, optional
            Mass of the impactor nose [kg].
        mass_trolley : float, optional
            Mass of the trolley [kg].

        Returns
        -------
        np.ndarray
            Acceleration corrected force.
        ----------
        """

        masses = {
            key: float(value.replace(',', '.').split()[0])
            for key, value in self.meta.items()
            if key.startswith('Mass')
        }

        mass_load_cell = (
            mass_load_cell
            if mass_load_cell is not None
            else masses.get("Mass Load Cell", 0.0)
        )

        mass_load_cell_after_gauge = (
            mass_load_cell_after_gauge
            if mass_load_cell_after_gauge is not None
            else masses.get("Mass Load Cell After Strain Gauge", 0.0)
        )

        mass_nose = (
            mass_nose
            if mass_nose is not None
            else masses.get("Mass Nose", 0.0)
        )

        mass_trolley = (
            mass_trolley
            if mass_trolley is not None
            else masses.get("Mass Trolley", 0.0)
        )

        mass_behind = (
            mass_trolley + 
            mass_load_cell - 
            mass_load_cell_after_gauge
        )

        mass_nose = (
            mass_nose + 
            mass_load_cell_after_gauge
        )

        force_corrected = (
            self._force * (
                1.0 + mass_nose / mass_behind
            )
        )

        return force_corrected
    
    @property
    def smoothing(self):
        return self._smoothing
    
    @smoothing.setter
    def smoothing(self, func):

        if func is not None and not callable(func):
            raise TypeError(
                "smoothing must be a callable or None"
            )
        
        self._smoothing = func
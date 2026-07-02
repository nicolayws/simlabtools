import numpy as np
from .ReadMesh import Mesh
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import re

class ECorrReader():
    """
    Reads eCorr .eco files from a complete DIC analysis.


    """

    def __init__(
            self,
            filepath: str | Path
    ):
        
        self.filepath = Path(filepath)

        self.name = self.filepath.stem

        self.meshes = []

        self.node1 = None
        self.node2 = None

        self._force: np.ndarray | None = None
        self._strain: np.ndarray | None = None
        self._stress: np.ndarray | None = None

        self.fracture_idx: int | None = None

        self.read()

    def read(self):

        ecofiles = sorted(
            self.filepath.glob("*.eco")
        )

        if len(ecofiles) == 0:
            raise RuntimeError(
                "Run eCorr DIC before trying to load .eco files"
            )

        for f in tqdm(
            ecofiles,
            desc=f"|=| {self.name} |=| Reading .eco files"
        ):
            
            self.meshes.append(
                Mesh(f)
            )
        
        self.image_numbers = self._read_image_numbers()
        
    def select_extensometer_nodes(
            self,
            img: np.ndarray | None = None
    ):
        
        mesh = self.meshes[0]

        fig, ax = plt.subplots()

        polys = []

        for elm in mesh.elm:

            nodes = elm.astype(int)-1

            nodes[2], nodes[3] = (
                nodes[3],
                nodes[2]
            )

            coords = mesh.nloc[nodes]

            polys.append(coords)
        
        pc = PolyCollection(
            polys, 
            edgecolors="black",
            facecolors="none",
            linewidth=0.5
        )

        ax.add_collection(pc)

        if img is not None:
            ax.imshow(img)
        
        ax.autoscale()

        ax.set_aspect("equal")
        ax.invert_yaxis()

        ax.set_title(
            "Choose two nodes for virtual extensometer"
        )

        selected_nodes = []
        selected_points = []

        def onclick(event):
            if event.inaxes != ax:
                return
            
            click = np.array([event.xdata, event.ydata])

            dist = np.linalg.norm(
                mesh.nloc - click,
                axis=1
            )

            node = np.argmin(dist)

            selected_nodes.append(node)
            selected_points.append(mesh.nloc[node])

            print(
                f"Valg {len(selected_nodes)}:"
                f" node {node}"
                f" ({mesh.nloc[node,0]:.1f},"
                f" {mesh.nloc[node,1]:.1f})"
            )

            ax.scatter(
                mesh.nloc[node, 0],
                mesh.nloc[node, 1],
                color="red",
                s=50,
                zorder=10
            )

            if len(selected_points) == 2:

                p1 = selected_points[0]
                p2 = selected_points[1]

                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    "r-",
                    lw=2
                )

                fig.canvas.draw()

                print(
                    f"\nValgte noder:"
                    f" {selected_nodes[0]} og {selected_nodes[1]}"
                )

                plt.pause(1.0)
                plt.close(fig)
            
            else:
                fig.canvas.draw()
        
        fig.canvas.mpl_connect(
            "button_press_event",
            onclick
        )

        plt.show()

        self.node1 = selected_nodes[0]
        self.node2 = selected_nodes[1]

        return self.node1, self.node2
    
    @staticmethod
    def _distance2D(coords1, coords2):

        return np.linalg.norm(
            coords2 - coords1
        )
    
    def compute_strain(self):

        if self.node1 is None:
            raise RuntimeError(
                "Select extensometer nodes first"
            )
        
        deform = []

        for mesh in self.meshes:

            pos1 = (
                mesh.nloc[self.node1] + 
                mesh.ndef[self.node1]
            )

            pos2 = (
                mesh.nloc[self.node2] + 
                mesh.ndef[self.node2]
            )

            deform.append(
                self._distance2D(
                    pos1,
                    pos2
                )
            )

        deform = np.asarray(deform)

        self.gauge_length = deform[0]

        self._strain = (
            deform - self.gauge_length
        ) / self.gauge_length

    def _read_image_numbers(self):

        with open(
            self.filepath / "input.txt", "r"
        ) as f:
            
            text = f.read()

        match = re.search(
            r"\*ImageNumbers\s+(\d+):(\d+)",
            text
        )

        if match is None:

            raise ValueError(
                "Coult not find image numbers"
            )
        
        start = int(match.group(1))
        stop = int(match.group(2))

        return np.arange(start, stop+1)

    def compute_stress(
            self,
            area: float
    ):
        
        logfile = sorted(
            self.filepath.glob('*Log.txt')
        )[0]

        log = np.genfromtxt(
            logfile,
            skip_header=5
        )

        force_all = log[:,1]

        self._force = (
            force_all[self.image_numbers - 2]
            * 1000
        )

        self._stress = (
            self._force / area
        )

    def compute_stress_strain(
            self,
            area: float,
            img: np.ndarray | None = None
    ):
        
        if self.node1 is None:
            self.select_extensometer_nodes(img)

        self.compute_strain()

        self.compute_stress(area)
    
    @property
    def first_image(
            self,
    ):

        fname = str.capitalize(
            self.meshes[0].filepath.stem
        ) + ".png"

        imgfolder = str.rsplit(
            fname,
            "_",
            1
        )[0]

        imgpath = Path(
            self.filepath / 
            imgfolder / 
            fname
        )

        img = plt.imread(
            imgpath
        )

        return img
    
    def summar(self):

        return {
            "name": self.name,
            "n_frames": len(self.meshes),
            "gauge_length_px": self.gauge_length,
            "max_strain": np.max(self.strain),
            "max_stress_MPa": np.max(self.stress)
        }
    
    @property
    def strain(self) -> np.ndarray:

        if self._strain is None:
            raise RuntimeError(
                "Run compute_stress_strain() first"
            )

        return self._strain

    @property
    def stress(self) -> np.ndarray:

        if self._stress is None:
            raise RuntimeError(
                "Run compute_stress_strain() first"
            )

        return self._stress

    @property
    def force(self) -> np.ndarray:

        if self._force is None:
            raise RuntimeError(
                "Run compute_stress_strain() first"
            )

        return self._force

    def find_fracture(self):
        """
        Find fracture based on force drop
        """
        
        raise RuntimeError("Not yet implemented")
    

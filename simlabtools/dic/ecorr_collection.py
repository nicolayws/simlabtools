from pathlib import Path
from .ecorr_reader import ECorrReader
from typing import Literal
import numpy as np

class ECorrCollection:

    def __init__(
            self,
            tests: list[ECorrReader] | None = None
    ):
        
        if tests is None:
            self.tests = []
        else:
            self.tests = tests

        self.mesh: Literal[
            "unique",
            "equal"
        ] = "unique"

    def add(
            self,
            test: ECorrReader
    ):
        
        self.tests.append(test)
        self.tests.sort(key=lambda x: x.name)

    def remove(
            self,
            name: str
    ):
        
        self.tests = [
            test for test in self.tests
            if test.name != name
        ]

    def __iter__(self):

        return iter(self.tests)
    
    def __len__(self):

        return len(self.tests)
    
    def __getitem__(self, idx):

        return self.tests[idx]

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(n_tests={len(self)})"
        )
    
    def __str__(self):

        out = [
            f"{self.__class__.__name__}",
            "-" * len(self.__class__.__name__),
            f"Number of tests: {len(self)}",
            ""
        ]

        for i, test in enumerate(self.tests, start=1):
            out.append(
                f"{i:>2}. {test.name}"
            )

        return "\n".join(out)
    
    @classmethod
    def from_directory(
        cls,
        path: str | Path
    ):
        
        tests = []

        for folder in Path(path).iterdir():

            if folder.is_dir():

                try:
                    tests.append(
                        ECorrReader(folder)
                    )
                except RuntimeError:
                    pass
        
        return cls(tests)

    def compute_stress_strain(
            self,
            jsonpath: str | Path,
            use_img: bool = False
    ):
        """
        Computes stress-strain for all tests in collection.

        Parameters
        ----------
        json: str | Path
            path to .json file containing cross-section measurements
        """
        import json

        with open(jsonpath, "r") as f:
            cross_sections = json.load(f)

        for it, test in enumerate(self):

            cross_section = cross_sections[test.name]
            area = (
                cross_section["w"] *
                cross_section["t"]
            )
            print(area)

            if self.mesh == "equal" and it != 0:
                node1 = self[0].node1
                node2 = self[0].node2

                test.node1 = node1
                test.node2 = node2

            if use_img:
                test.compute_stress_strain(
                    area=area,
                    img=test.first_image
                )
            else:
                test.compute_stress_strain(
                    area=area,
                    img=None
                )
    
    def plot_stress_strain(
            self,
            ax=None
    ):
        
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        for test in self:

            ax.plot(
                test.strain,
                test.stress,
                label=test.name
            )
        
        ax.set_xlabel(
            "Engineering strain [-]"
        )
        ax.set_ylabel(
            "Engineering stress [MPa]"
        )
        ax.legend()

        return ax

    def save_txt(
            self,
            fpath: str | Path = "stress_strain_collection.csv"
    ):
        """
        Save stress-strain data for all tests to a CSV file.

        The output file contains two columns for each test:

            {test_name}_strain
            {test_name}_stress

        Tests with different numbers of data points are automatically
        padded with NaN values.

        Parameters
        ----------
        fpath : str or Path, optional
            Path to output CSV file.

        Examples
        --------
        >>> collection.save("stress_strain_collection.csv")

        The data can later be loaded with NumPy:

        >>> data = np.genfromtxt(
        ...     "stress_strain_collection.csv",
        ...     delimiter=",",
        ...     names=True
        ... )
        >>> strain = data["Test01_strain"]
        >>> stress = data["Test01_stress"]
        """

        fpath = Path(fpath)

        if len(self) == 0:
            raise RuntimeError(
                "Cannot save an empty ECorrCollection."
            )

        data = {}

        for test in self:

            if not hasattr(test, "stress"):
                raise RuntimeError(
                    f"Stress-strain has not been computed "
                    f"for test '{test.name}'."
                )

            data[f"{test.name}_strain"] = np.asarray(
                test.strain
            )
            data[f"{test.name}_stress"] = np.asarray(
                test.stress
            )

        max_len = max(
            len(v) for v in data.values()
        )

        out = {}

        for key, values in data.items():

            arr = np.full(max_len, np.nan)
            arr[:len(values)] = values
            out[key] = arr

        header = ",".join(out.keys())

        np.savetxt(
            fpath,
            np.column_stack(list(out.values())),
            delimiter=",",
            header=header,
            comments=""
        )

    def save_npz(
            self,
            fpath: str | Path = "stress_strain_collection.npz"
    ):
        """
        Save stress-strain data for all tests to a compressed NumPy archive.

        Each test is stored as separate arrays using the keys

            {test_name}_strain
            {test_name}_stress

        Additional quantities, such as force and displacement, are saved
        if available.

        Parameters
        ----------
        fpath : str or Path, optional
            Path to output `.npz` file.

        Raises
        ------
        RuntimeError
            If the collection is empty or if stress-strain data have not
            been computed for one or more tests.

        Examples
        --------
        Save the collection:

        >>> collection.save_npz()

        Load the data:

        >>> import numpy as np
        >>> data = np.load("stress_strain_collection.npz")
        >>> strain = data["Test01_strain"]
        >>> stress = data["Test01_stress"]

        List available arrays:

        >>> print(data.files)
        """

        fpath = Path(fpath)

        if len(self) == 0:
            raise RuntimeError(
                "Cannot save an empty ECorrCollection."
            )

        data = {}

        for test in self:

            if test._stress is None or test._strain is None:
                raise RuntimeError(
                    f"Stress-strain has not been computed "
                    f"for test '{test.name}'."
                )

            data[f"{test.name}_strain"] = test.strain
            data[f"{test.name}_stress"] = test.stress

            if test._force is not None:
                data[f"{test.name}_force"] = test.force

            if test.node1 is not None:
                data[f"{test.name}_node1"] = np.asarray(test.node1)

            if test.node2 is not None:
                data[f"{test.name}_node2"] = np.asarray(test.node2)

            if test.fracture_idx is not None:
                data[f"{test.name}_fracture_idx"] = np.asarray(
                    test.fracture_idx
                )

        np.savez_compressed(
            fpath,
            **data
        )

    @classmethod
    def load_npz(
            cls,
            fpath: str | Path
    ):
        """
        Load stress-strain data from a NumPy archive.

        Parameters
        ----------
        fpath : str or Path
            Path to `.npz` file created with :meth:`save_npz`.

        Returns
        -------
        ECorrCollection
            Collection containing the stored tests.

        Examples
        --------
        >>> collection = ECorrCollection.load_npz(
        ...     "stress_strain_collection.npz"
        ... )

        >>> collection[0].strain
        >>> collection[0].stress
        """

        from types import SimpleNamespace

        data = np.load(fpath)

        names = sorted({
            key.removesuffix("_strain")
            for key in data.files
            if key.endswith("_strain")
        })

        tests = []

        for name in names:

            test = ECorrReader.__new__(ECorrReader)

            test.filepath = None  # type: ignore
            test.name = name
            test.meshes = []

            test._strain = data[f"{name}_strain"]
            test._stress = data[f"{name}_stress"]

            test._force = (
                data[f"{name}_force"]
                if f"{name}_force" in data
                else None
            )

            test.node1 = (
                int(data[f"{name}_node1"])
                if f"{name}_node1" in data
                else None
            )

            test.node2 = (
                int(data[f"{name}_node2"])
                if f"{name}_node2" in data
                else None
            )

            test.fracture_idx = (
                int(data[f"{name}_fracture_idx"])
                if f"{name}_fracture_idx" in data
                else None
            )

            tests.append(test)

        return cls(tests)
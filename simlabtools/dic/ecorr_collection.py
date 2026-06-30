from pathlib import Path
from .ecorr_reader import ECorrReader
from typing import Literal

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
    

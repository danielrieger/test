import IMP
import IMP.core
import IMP.algebra
import IMP.atom

import smlm_score.utility
from smlm_score.utility.data_handling import *

import numpy as np
import pandas as pd


class Model:
    """Top-level model container for the SMLM-IMP pipeline.

    Manages SMLM data, PDB structure, and restraint objects.
    """
    def __init__(self, smlm_data_path=None, pdb_data_path=None, scoring_restraint=None,
                 imp_restraint=None, brownian_restraint=None):
        """
        Initialize the Model with setup and optional data paths and restraints.

        Args:
            smlm_data_path (str): Path to SMLM localization data.
            pdb_data_path (str): Path to PDB structure file.
            scoring_restraint: Scoring restraint object.
            imp_restraint: IMP restraint object.
            brownian_restraint: Brownian dynamics restraint object.
        """

        print("model.py - --init--")
        self.model = None
        # data
        self.smlm_data_path = smlm_data_path
        self.data_xyz = None

        self.pdb_data_path = pdb_data_path
        self.p_root = None
        self.h_root = None
        self.hier = None
        self.chains = None
        self.av_parameter = None


        # restraints
        self.scoring_restraint = scoring_restraint
        self.imp_restraint = imp_restraint
        self.brownian_restraint = brownian_restraint


    def initialize(self) -> None:
        """
        Initialize the model with loaded data and restraints.
        """
        print("model.py - initialize")
        self.model = IMP.Model()
        self.p_root = IMP.Particle(self.model, "root")
        self.h_root = IMP.atom.Hierarchy.setup_particle(self.p_root)


        # Load SMLM data if path is provided
        if self.smlm_data_path:
            print(f"Loading SMLM data from {self.smlm_data_path}")
            self.data_xyz = pd.read_csv(self.smlm_data_path, delimiter=',')
            self.data_xyz['precision'] = 1. / np.sqrt(self.data_xyz['Amplitude_0_0'])
            self.data_xyz['variance'] = self.data_xyz['precision'] ** 2.0
            sigma = np.array(self.data_xyz['precision'])

        # Load PDB data if path is provided
        if self.pdb_data_path:
            print(f"Loading PDB data from {self.pdb_data_path}")
            self.hier = IMP.atom.read_mmcif(self.pdb_data_path, self.model)
            self.chains = ['0', '0-2', '0-3', '0-4', '0-5', '0-6', '0-7', '0-8']
            self.av_parameter = {
                "linker_length": 150.0,
                "radii": (30.0, 0.0, 0.0),
                "linker_width": 4.0,
                "allowed_sphere_radius": 10.0,
                "contact_volume_thickness": 0.0,
                "contact_volume_trapped_fraction": -1,
                "simulation_grid_resolution": 8.0
            }

        # Initialize restraints
        print("Initializing restraints...")


    def filterSMLM(self):
        """Apply spatial filtering to the loaded SMLM data."""
        print("model.py - filterSMLM")
        self.data_xyz, *_ = flexible_filter_smlm_data(self.data_xyz)


    def run(self):
        """Execute the model simulation or optimization process."""
        print("model.py - run")
        print("Running the model...")

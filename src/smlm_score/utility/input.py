
import json
import pandas as pd
import numpy as np


def read_experimental_data(smlm_data_path):
    """
    Load and process SMLM localization data from a CSV file.

    Parameters
    ----------
    smlm_data_path : str
        Path to the CSV file containing SMLM localizations.
        Expected columns: 'x [nm]', 'y [nm]', 'Amplitude_0_0'.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with added 'precision' and 'variance' columns,
        or None if loading fails.
    """
    print(f"Loading SMLM data from {smlm_data_path}")

    try:
        data_xyz = pd.read_csv(smlm_data_path, delimiter=',')

        # `precision` is treated as a 1-sigma proxy derived from amplitude.
        # The scoring code consumes variances/covariances, so expose both.
        data_xyz['precision'] = 1. / np.sqrt(data_xyz['Amplitude_0_0'])
        data_xyz['variance'] = data_xyz['precision'] ** 2.0

        return data_xyz

    except FileNotFoundError:
        print(f"File not found at path: {smlm_data_path}")
        return None
    except KeyError:
        print("Error: 'Amplitude_0_0' column not found in the data.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def read_parameters_from_json(file_path):
    """
    Read AV and chain parameters from a JSON configuration file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing the parameters.
        Expected keys: 'chains', 'residue_index', 'atom_name', 'av_parameter'.

    Returns
    -------
    dict or None
        Dictionary containing all parameters with 'radii' converted to tuple,
        or None if loading fails.
    """
    try:
        with open(file_path, 'r') as file:
            parameters = json.load(file)

            if parameters:
                chains = parameters['chains']
                residue_index = parameters['residue_index']
                atom_name = parameters['atom_name']
                av_parameter = parameters['av_parameter']
                av_parameter['radii'] = tuple(av_parameter['radii'])
            return parameters

    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

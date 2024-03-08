import os
import argparse
import numpy as np

def file_loader():
    parser = argparse.ArgumentParser(description = "Loading and printing an array")
    parser.add_argument("--input", "-i")
    args = parser.parse_args()
    return args

def main():
    args = file_loader()
    filename = os.path.join(
                            "..",
                            "..",
                            "..",
                            "cds-vis-data",
                            "data",
                            "sample-data",
                            args.input      # take one input argument given by the user in the command line
                            )
    data = np.loadtxt(filename, delimiter = ",")
    print(data)

# OBS: this filepath must be manually changed depending on where the .py script is executed
# (remove ".." if its executed from the command line - otherwise dont do anything)

if __name__=="__main__":
    main()

# To execude the code in the command line (in the right directory):
    # $ python test_script_session6.py --input "sample-data-01.csv"
    # $ python test_script_session6.py -i "sample-data-01.csv"

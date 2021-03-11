#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2019 De Montfort University. All rights reserved.
#
#

"""
Helper script to copy test.info.txt files to corresponding results folders.
The MA uses the fact that the solutions are symmetrical and rotates solutions
to complete symmetrical board sizes. This means that there is only one
test.info.txt file.

Command line usage example:
    py -3 copy_results.py
"""

import os
import sys
import argparse
import random
import time                     # Used to time script execution.
import re

import common as cmn
import results

__author__ = 'David Kind'
__date__ = '29-10-2019'
__version__ = '1.0'
__copyright__ = 'Copyright(c) 2019 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)


def main():
    """
    Ensure all the MA results have a copy of the results text file.
    Ref: common.py for file and directory definitions.
    :param: n/a
    :return: n/a
    """
    # Create a list of all the folders that contain a results file.
    results_folders = results.check_results_folder(cmn.MA_FOLDER)
    # Extract the dimensions; use a regex
    regex_mxn = re.compile(r"(\d+)x(\d+)", re.IGNORECASE)
    # Now loop through all the results folders and copy across results files to
    # symmetrical results folders.
    for result_folder in results_folders:
        # Extract the board dimensions
        mxn_tuple = re.findall(regex_mxn, result_folder)
        # Note: we don't process folders where m == n
        if mxn_tuple[0][0] == mxn_tuple[0][1]:
            continue
        # Need to break out the path and the filename
        old = "{}x{}".format(mxn_tuple[0][0], mxn_tuple[0][1])
        new = "{}x{}".format(mxn_tuple[0][1], mxn_tuple[0][0])
        symfolder = result_folder.replace(old, new)
        # Is the symmetrical folder already listed as having results?
        # If so then we can skip it.
        if symfolder in results_folders:
            continue
        # Now copy the results file from the source folder to its symmetrical
        # board folder.
        src = os.path.join(result_folder, cmn.RESULTS_FILE)
        dst = os.path.join(symfolder, cmn.RESULTS_FILE)
        # Note: we have to modify the contents to reflect the board dimensions.
        with open(src, "r") as fin:
            contents = fin.read()
        # Now update the board dimensions and write it out again.
        contents = contents.replace(old, new)
        with open(dst, "w") as fout:
            fout.write(contents)
        print("Copied {} from {} to {}".format(cmn.RESULTS_FILE, result_folder, symfolder))
    print("Done")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('--version', action='version', version=SCRIPTINFO)

    # Start up the application
    print("Running: {}".format(SCRIPTINFO))
    main()

# EOF

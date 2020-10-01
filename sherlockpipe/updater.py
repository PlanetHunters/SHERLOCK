import os
import shutil
import time
from argparse import ArgumentParser

import eleanor
from sherlockpipe.vet import Vetter
from sherlockpipe.ois.OisManager import OisManager


if __name__ == '__main__':
    ap = ArgumentParser(description='Updater of SHERLOCK PIPEline metadata')
    ap.add_argument('--clean', dest='clean', action='store_true', help="Whether to remove all data and download it again.")
    ap.add_argument('--only_ois', dest='ois', action='store_true', help="Whether to only refresh objects of interest.")
    ap.add_argument('--force', dest='force', action='store_true', help="Whether to ignore update timestamps and do everything again.")
    args = ap.parse_args()
    ois_manager = OisManager()
    timestamp_ois_path = os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_ois.txt')
    timestamp_eleanor_path = os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_eleanor.txt')
    timestamp_latte_path = os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_latte.txt')
    ois_timestamp = 0
    eleanor_timestamp = 0
    latte_timestamp = 0
    if os.path.exists(timestamp_ois_path):
        with open(timestamp_ois_path, 'r+') as f:
            ois_timestamp = f.read()
    if os.path.exists(timestamp_eleanor_path):
        with open(timestamp_eleanor_path, 'r+') as f:
            eleanor_timestamp = f.read()
    if os.path.exists(timestamp_latte_path):
        with open(timestamp_latte_path, 'r+') as f:
            latte_timestamp = f.read()
    if args.force or time.time() - float(ois_timestamp) > 3600 * 24 * 7:
        print("------------------ Reloading TOIs ------------------")
        ois_manager.update_tic_csvs()
        print("------------------ Reloading KOIs ------------------")
        ois_manager.update_kic_csvs()
        print("------------------ Reloading EPICs ------------------")
        ois_manager.update_epic_csvs()
        with open(os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_ois.txt'), 'w+') as f:
            f.write(str(time.time()))
    if (args.force or time.time() - float(eleanor_timestamp) > 3600 * 24 * 7) and not args.ois:
        print("------------------ Reloading ELEANOR TESS FFI data ------------------")
        eleanorpath = os.path.join(os.path.expanduser('~'), '.eleanor')
        eleanormetadata = eleanorpath + "/metadata"
        if args.clean and os.path.exists(eleanorpath) and os.path.exists(eleanormetadata):
            shutil.rmtree(eleanormetadata, ignore_errors=True)
        if not os.path.exists(eleanorpath):
            os.mkdir(eleanorpath)
        if not os.path.exists(eleanormetadata):
            os.mkdir(eleanormetadata)
        for sector in range(1, 52):
            sectorpath = eleanorpath + '/metadata/s{:04d}'.format(sector)
            if not os.path.exists(sectorpath) or not os.path.isdir(sectorpath) or not os.listdir(sectorpath):
                try:
                    eleanor.Update(sector)
                except:
                    os.rmdir(sectorpath)
        with open(os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_eleanor.txt'), 'w+') as f:
            f.write(str(time.time()))
    if (args.force or time.time() - float(latte_timestamp) > 3600 * 24 * 7) and not args.ois:
        print("------------------ Reloading LATTE data ------------------")
        Vetter(None).update()
        with open(os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_latte.txt'), 'w+') as f:
            f.write(str(time.time()))
    print("DONE")
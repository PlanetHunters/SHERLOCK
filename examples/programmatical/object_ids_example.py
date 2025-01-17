from contextlib import contextmanager
from timeit import default_timer
from sherlockpipe.search.sherlock import Sherlock
from lcbuilder.objectinfo.InputObjectInfo import InputObjectInfo
from lcbuilder.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo

from sherlockpipe.search.sherlock_target import SherlockTarget


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: str(default_timer() - start)
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: str(end - start)


with elapsed_timer() as elapsed:
    # Adding several kinds of objects to the run: one short cadence TIC, one FFI TIC, one coordinates FFI, one input
    # file related to a TIC and one plain input file.
    # Ensure that your input light curve CSV files have three columns: #TBJD,flux,flux_err
    sherlock = Sherlock([SherlockTarget(MissionObjectInfo(mission_id="TIC 181804752", sectors='all')),
                                        SherlockTarget(MissionObjectInfo(mission_id="TIC 259168516", sectors=[14, 15])),
                                        #SherlockTarget(MissionObjectInfo(ra=14, dec=19, sectors='all')),
                                        SherlockTarget(MissionInputObjectInfo(mission_id="TIC 470381900", input_file="example_lightcurve.csv")),
                                        SherlockTarget(InputObjectInfo("example_lc.csv"))])\
        .run()
    print("Analysis took " + elapsed() + "s")

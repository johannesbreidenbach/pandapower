# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

# imports from pandapower
import pandapower.networks as pn
from pandapower.run import runpp

if __name__ == '__main__':

    net14 = pn.case14()
    runpp(net14)

    net30 = pn.case30()
    runpp(net30)

    print(f"whats up")
TITLE ampa synapse 

NEURON {
    POINT_PROCESS ampa
    NONSPECIFIC_CURRENT i
    RANGE g, a, b, e, gMax, DA
}

UNITS {
    (uS) = (microsiemens)
    (nA) = (nanoamp)
    (mV) = (millivolt)
}

PARAMETER {
    tauRise = 0.55  (ms)
    tauFall = 2.2  (ms)
    e = 0  (mV)
    gMax = 1
    DA = 0
}

ASSIGNED {
    v  (mV)
    i  (nA)
    g  (uS)
    factor
}

INITIAL { 
    a = 0  
    b = 0 
    factor = tauRise*tauFall / (tauFall-tauRise)
}

STATE {
    a (uS)
    b (uS)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = (b-a)*factor
    i = (gMax*(1-0.2*DA))*g*(v-e)
}

DERIVATIVE states {
    a' = -a/tauRise
    b' = -b/tauFall
}

NET_RECEIVE(weight (uS)) {
    a = a + weight
    b = b + weight
}
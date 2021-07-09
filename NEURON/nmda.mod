TITLE nmda synapse 

NEURON {
    POINT_PROCESS nmda
    NONSPECIFIC_CURRENT i
    RANGE g, a, b, e, s, gMax, DA
}

UNITS {
    (uS) = (microsiemens)
    (nA) = (nanoamp)
    (mV) = (millivolt)
}

PARAMETER {
    tauRise = 10.6  (ms)
    tauFall = 285.0  (ms)
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
    LOCAL s
    SOLVE states METHOD cnexp
    : SOLVE states METHOD derivimplicit
    g = (b-a)*factor
    s = 1.50265 / (1+0.33*exp(-0.0625*v))
    i = (gMax*(1+0.4*DA))*g*s*(v-e)
}

DERIVATIVE states {
    a' = -a/tauRise
    b' = -b/tauFall
}

NET_RECEIVE(weight (uS)) {
    a = a + weight
    b = b + weight
}
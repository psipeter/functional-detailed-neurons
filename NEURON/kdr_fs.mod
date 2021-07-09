: Delayed rectifier K+ channel for Fast Spiking Interneuron

NEURON {
	SUFFIX kdr_fs
	USEION k READ ki, ko WRITE ik
	RANGE gkdrbar, ik, gk
	
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}
PARAMETER {
	v (mV)
	dt (ms)
	gkdrbar= 0.0080 (mho/cm2) <0,1e9>
	
	
}

STATE {
	n
}

ASSIGNED {
	ik (mA/cm2)
	inf
	tau (ms)
	gk (mho/cm2)
	ek (mV)
	ki (mM)
	ko (mM)

}


INITIAL {
	rate(v)
	n = inf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gk= gkdrbar*n*n*n*n
	ek = 25 * log(ko/ki)
	ik = gk*(v-ek)
	
}

DERIVATIVE states {
	rate(v)
	n' = (inf-n)/tau
}

UNITSOFF

FUNCTION alf(v){
	alf = 0.3 * exp((v + 35)/10.67)
}


FUNCTION bet(v) {
	bet = 0.3 * exp(-(v + 35)/42.68)
}	



PROCEDURE rate(v (mV)) {LOCAL q10, sum, aa, ab
	
	aa=alf(v) ab=bet(v) 
	
	sum = aa+ab
	inf = aa/sum
	tau = 1/(sum)
	
	
}

UNITSON	




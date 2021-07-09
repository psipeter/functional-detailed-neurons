: Fast Na+ channel for Fast Spiking Interneurons

NEURON {
	SUFFIX Naf_fs
	USEION na READ ena WRITE ina
	RANGE gnafbar, ina, gna
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

PARAMETER {
	v (mV)
	dt (ms)
	gnafbar= 0.100 (mho/cm2) <0,1e9>
	ena = 55 (mV)

}
STATE {
	m h
}
ASSIGNED {
	ina (mA/cm2)
	minf hinf 
	mtau (ms)
	htau (ms)
	gna (mho/cm2)
	
}



INITIAL {
	rate(v)
	m = minf
	h = hinf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gna = gnafbar*m*m*m*h
	ina = gna*(v-55)
	
}

DERIVATIVE states {
	rate(v)
	m' = (minf-m)/mtau
	h' = (hinf-h)/htau
}

UNITSOFF

FUNCTION malf( v){ 
	malf = 4.2 * exp((v + 34.5)/11.57)
}


FUNCTION mbet(v(mV))(/ms) {
	mbet = 4.2 * exp(-(v + 34.5)/27)
}	


FUNCTION half(v(mV))(/ms) {
	half = 0.09 * exp(-(v + 45)/33)
}


FUNCTION hbet(v(mV))(/ms) {
	hbet = 0.09 * exp((v + 45)/12.2)
}




PROCEDURE rate(v (mV)) {LOCAL q10, msum, hsum, ma, mb, ha, hb
	

	ma=malf(v) mb=mbet(v) ha=half(v) hb=hbet(v)
	
	msum = ma+mb
	minf = ma/msum
	mtau = 1/(msum)
	
	
	hsum=ha+hb
	hinf=ha/hsum
	htau = 1 / (hsum)
	
}

	
UNITSON



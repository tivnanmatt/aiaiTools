
import os
from scipy import io
from scipy import interpolate
import numpy as np



def sourceSpectrum( 	kvp, 
			Emin=1.5, 
			Emax=150.5, 
			nEnergy=150 ):

        # INPUTS:
        #       kvp                             kvp of the xray source
        #       Emin                            energy in keV of minimum energy bin (inclusive)
        #       Emax                            energy in keV of maximum energy bin (inclusine)
        #       nEnergy                         number of energy bins
        #       
        # OUTPUTS:
        #       sourceSpectrum                  source spectrum

        kvp = np.array(kvp)

        Emin = np.array(Emin).astype('float')
        Emax = np.array(Emax).astype('float')
        nEnergy = np.array(nEnergy).astype('float')
        dE = (Emax - Emin)/ ( nEnergy - 1.0 )
        E = np.arange(0,nEnergy.astype('int')).astype('float')*dE + Emin
        E_table = np.arange(0,150).astype('float')*1.0 + 1.5

        # load in the SPEKTR attenuation tables
        spTASMICS = io.loadmat(os.path.dirname(__file__) + '/' + 'spektrTASMICSdata.mat')['spTASMICS']
	

        kvp1 = np.floor(kvp)
        kvp2 = np.ceil(kvp)

        sourceSpectrum1 = np.interp(E, E_table, spTASMICS[:,(kvp1-20).astype('int')])
        sourceSpectrum2 = np.interp(E, E_table, spTASMICS[:,(kvp2-20).astype('int')])
        

        if kvp1 == kvp2:
                sourceSpectrum = sourceSpectrum1
        else:
	        sourceSpectrum = sourceSpectrum1 + ((kvp - kvp1)/(kvp2-kvp1))*sourceSpectrum2

        return sourceSpectrum


def sensitivitySpectrum_CsI(	L_CsI, 
				Emin=1.5, 
				Emax=150.5, 
				nEnergy=150):

        # INPUTS:
        #       L_CsI                           thickness of CsI scintilator
        #       Emin                            energy in keV of minimum energy bin (inclusive)
        #       Emax                            energy in keV of maximum energy bin (inclusine)
        #       nEnergy                         number of energy bins
        #       
        # OUTPUTS:
        #       systemSensitivity               source spectrum

        L_CsI = np.array(L_CsI)

        Emin = np.array(Emin).astype('float')
        Emax = np.array(Emax).astype('float')
        nEnergy = np.array(nEnergy).astype('float')
        dE = (Emax - Emin)/ ( nEnergy - 1.0 )
        E = np.arange(0,nEnergy.astype('int')).astype('float')*dE + Emin
	
        rho_CsI = 4.51 / 1000.0 # g/mm^3
	
        muPerRho_CsI = mass_attenuation_spectrum([53, 55], Emin=Emin, Emax=Emax, nEnergy=nEnergy)
	
        interactionSpectrum = (1.0 - np.exp(-L_CsI*rho_CsI*muPerRho_CsI))

        # work function (visible photons / kev)
        W = 55.6

        # optical coupling (typical value for FPD)
        g4 = 0.59

        conversionSpectrum = W * E

        gain = g4

        return (interactionSpectrum, conversionSpectrum, gain)




def mass_attenuation_spectrum( 	atomicNumbers, 
				Emin=1.5, 
				Emax=150.5, 
				nEnergy=150 ):

        # INPUTS:
        #       atomicNumbers                   list of atomic numbers 1-91 for this material
        #                                       an element can appear more than once 
        #                                       e.g. Water should be atomicNumbers=[1, 1, 8]
        #       Emin                            energy in keV of minimum energy bin (inclusive)
        #       Emax                            energy in keV of maximum energy bin (inclusine)
        #       nEnergy                         number of energy bins
        #       
        # OUTPUTS:
        #       massAttenuationSpectrum         mass attenuation spectrum in mm^2/g 


        atomicNumbers = np.array(atomicNumbers)
        atomicNumbers.shape = np.size(atomicNumbers), 1

        Emin = np.array(Emin).astype('float')
        Emax = np.array(Emax).astype('float')
        nEnergy = np.array(nEnergy).astype('float')
        dE = (Emax - Emin)/ ( nEnergy - 1.0 )
        E = np.arange(0,nEnergy.astype('int')).astype('float')*dE + Emin
        E_table = np.arange(0,150).astype('float')*1.0 + 1.5

        # load in the SPEKTR attenuation tables
        data = io.loadmat(os.path.dirname(__file__) + '/' + 'spektrMuRhoElements.mat')['A']
        muPerRhoElements = np.zeros([93,150],dtype='float')
        for iZ in np.arange(1,93):
                muPerRhoElements[iZ,:] = data[iZ-1][0][:,8]
        muPerRhoElements = muPerRhoElements*100.0 # conver cm^2/g to mm^2/g

        data = io.loadmat(os.path.dirname(__file__) + '/' + 'spektrAtomicMassElements.mat')['AMU']
        amu = np.zeros([93,1],dtype='float')
        for iZ in np.arange(1,93):
                amu[iZ,:] = data[iZ-1, 2]

        sumAmu = 0
        mu = 0
        for an in atomicNumbers:
                sumAmu += amu[an]
                mu += amu[an]*muPerRhoElements[an, :].reshape(np.shape(E_table))
        mu /= sumAmu

        f = interpolate.interp1d(E_table, mu, 'linear',bounds_error=False,fill_value='extrapolate')

        mu = f(E)

        mu = np.squeeze(mu)

        return mu





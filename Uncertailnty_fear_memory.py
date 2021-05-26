import scipy as sp 
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

class Basic_Neural_Circuit_():

    def __init__(self):
        self.alpha_F    =  .4
        self.alpha_P    =  .4
        self.alpha_E    =  .4
        self.Wfe    =   2
        self.dt =   .01
        self.T  =   .5
        self.t  =   sp.arange(0.0, 30,self.dt)
        self.cs_at  =   sp.arange(.5, 30, 1)
        self.p_us_at    =   np.array([.5, 2.5, 5.5 ,8.5 ,9.5, 12.5, 13.5, 14.5, 17.5, 19.5])
        self.us_at      =   sp.arange(.5, 10, 1)
        
        
    def rect(self,T):
        """create a centered rectangular pulse of width $T"""
        return lambda t: (-T/2 <= t) & (t < T/2)
 
    def pulse_train(self,t, at, shape):
        """create a train of pulses over $t at times $at and shape $shape"""
        return np.sum(shape(t - at[:,np.newaxis]), axis=0)


    def _ToExt(self,f,p,us,e):
        return f* (p-us-e)

    def _ToPer(self,p,us):
        return us-p

    def _ToFear(self,f,us):
        return us-f        

    
    def _activity(self):
        shape=  self.rect(self.T)
        cs= self.pulse_train(self.t,self.cs_at,shape)
        us= self.pulse_train(self.t,self.us_at,shape)
        
        P=  np.zeros(np.size(self.t))
        E=  np.zeros(np.size(self.t))
        F=  np.zeros(np.size(self.t))

        Wf= np.zeros(np.size(self.t))
        Wp= np.zeros(np.size(self.t))
        We= np.zeros(np.size(self.t))

        LToE=   np.zeros(np.size(self.t))
        LToP=   np.zeros(np.size(self.t))
        LToF=   np.zeros(np.size(self.t))

        dWf= dWp= dWe= 0


        for t_order, t_step in enumerate(self.t):

            if (t_order>0):

                if (cs[t_order] ==1):

                    Wp[t_order] =  Wp[t_order-1]+ self.dt*  dWp
                    We[t_order] =  We[t_order-1]+  self.dt* dWe
                    Wf[t_order] =  Wf[t_order-1]+  self.dt* dWf

                    P[t_order]= (Wp[t_order])* cs[t_order]
                    E[t_order]= (We[t_order])* cs[t_order]
                    F[t_order]= (Wf[t_order])* cs[t_order]- self.Wfe* E[t_order]

                    if ( (us[t_order]- F[t_order])>=0):
                        LToF[t_order]=  (us[t_order]- F[t_order])
                        dWf= self.alpha_F* cs[t_order]* (us[t_order]- F[t_order])
                        
                        "print ('dwf', dWf)"
                    else:
                        dWf=0
                        

                    if ((us[t_order]-P[t_order]) >=0):

                        LToP[t_order]=  (us[t_order]- P[t_order])
                        dWp= self.alpha_P* cs[t_order]* (us[t_order]- P[t_order])
                        "print ('dwp', dWp)"

                    else:
                        dWp= 0
                    
                    if ((F[t_order]*(P[t_order]- us[t_order]- E[t_order])) >=0 ):

                        LToE[t_order]=  (F[t_order]*(P[t_order]- us[t_order]- E[t_order]))
                        dWe= self.alpha_E* cs[t_order]* (F[t_order]*( P[t_order]-us[t_order]- E[t_order]))
                        "print ('dwe', dWe)"

                    else:
                        dWe= 0

                    
                

                else:

                    P[t_order]= P[t_order-1]        
                    E[t_order]= E[t_order-1]        
                    F[t_order]= F[t_order-1]

                    Wf[t_order]= Wf[t_order-1]        
                    We[t_order]= We[t_order-1]        
                    Wp[t_order]= Wp[t_order-1]

                    LToF[t_order]=  LToF[t_order-1] 
                    LToE[t_order]=  LToE[t_order-1] 
                    LToP[t_order]=  LToP[t_order-1] 

                    print('P, E, F,\t', P[t_order],E[t_order],F[t_order])
                    print('cs, us,\t\t', cs[t_order],us[t_order])
                    print('Wp, We, Wf',Wp[t_order],We[t_order],Wf[t_order],'\n')
                 
                "print('dwp, dwe, dwf',dWp,dWe,dWf,'\n')"
        return self.t, F, E, P,cs, us, Wf,We, Wp, LToE, LToF, LToP

class Extended_Neural_Circuit_():

    def __init__(self):
        
        self.alpha_F=   .4
        self.alpha_P=   .4
        self.alpha_E1=  .4
        self.alpha_E2=  .4
        self._alphaE2=  .03   
        self.Wfe2=      2      
        self.We2e1=     .05
        self.beta_E2=   .01
        self._betaE2=   .005

        self.dt=    .01
        self.T=     .5
        self.t=     sp.arange(0.0, 70.5,self.dt)
        self.cs_at=  sp.arange(.5, 30, 1)
        self.rest_t=  sp.arange(30.5, 60.5, 1)
        self.retrive_t=  sp.arange(60.5, 70.5, 1)
        self.p_us_at= np.array([.5, 2.5, 5.5 ,8.5 ,9.5, 12.5, 13.5, 14.5, 17.5, 19.5])
        self.us_at= sp.arange(.5, 10, 1)
        self._vmPFC=    sp.arange(12.5,28.5,1)
    
        
        

    def rect(self,T):
        """create a centered rectangular pulse of width $T"""
        return lambda t: (-T/2 <= t) & (t < T/2)
 
    def pulse_train(self,t, at, shape):
        """create a train of pulses over $t at times $at and shape $shape"""
        return np.sum(shape(t - at[:,np.newaxis]), axis=0)

    def _rest(self, at):
            return 0

    def _ToExt(self,f,p,us,e):
        return f* (p-us-e)

    def _ToPer(self,p,us):
        return us-p

    def _ToFear(self,f,us):
        return us-f        

    def _Stimulus(self,t,T,st):

        shape= self.rect(T)
        
        return self.pulse_train(t,st,shape)




    def _activity(self):

        
        cs=self._Stimulus(self.t,self.T,self.cs_at)
        
        shape=  self.rect(self.T)
        act_shape=  self.rect(5)
        """
        cs1= self.pulse_train(self.t,self.cs_at,shape)
        "cs2= np.zeros(np.size(self.rest_t))"
        cs2= self._rest(self._rest)
        cs3= self.pulse_train(self.t,self.retrive_t,shape)
        cs= cs1+ cs2+ cs3
        """
        us= self.pulse_train(self.t,self.us_at,shape)
        vmpfc=  2* self.pulse_train(self.t,self._vmPFC,act_shape)
        """
        vmpfc[1000:1100]= np.arange(0,10,.1)
        vmpfc[2990:3100]= np.arange(10,0,-.1)
        vmpfc=  np.zeros(np.size(self.t))"
        vmpfc1= 2.5*(signal.sawtooth(2 * np.pi * 5 * self._vmPFC) + 1)
        vmpfc1= vmpfc+vmpfc1
        """
        P=  np.zeros(np.size(self.t))
        E1= np.zeros(np.size(self.t))
        E2= np.zeros(np.size(self.t))
        F=  np.zeros(np.size(self.t))

        Wf= np.zeros(np.size(self.t))
        Wp= np.zeros(np.size(self.t))
        We1=    np.zeros(np.size(self.t))
        We2=    np.zeros(np.size(self.t))
        _We2=    np.zeros(np.size(self.t))

        LToE1=   np.zeros(np.size(self.t))
        LToE2=   np.zeros(np.size(self.t))
        LToP=   np.zeros(np.size(self.t))
        LToF=   np.zeros(np.size(self.t))
        """
        Wf= np.array([])
        Wp= np.array([])
        We= np.array([])"""
        dWf= dWp= dWe1= dWe2= d_We2= 0


        for t_order, t_step in enumerate(self.t):

            if (t_order>0):

                if (cs[t_order]==1):

                    Wp[t_order] =   Wp[t_order-1]+  self.dt* dWp
                    We1[t_order] =  We1[t_order-1]+ self.dt*dWe1
                    We2[t_order] =  We2[t_order-1]+ self.dt* dWe2
                    Wf[t_order] =   Wf[t_order-1]+  self.dt*dWf
                    _We2[t_order] = _We2[t_order-1]+ self.dt* d_We2

                    P[t_order]= Wp[t_order]* cs[t_order]
                    E1[t_order]= We1[t_order]* cs[t_order]+ vmpfc[t_order]
                    E2[t_order]= We2[t_order]* cs[t_order]+ self.We2e1* E1[t_order] 
                    F[t_order]= Wf[t_order]* cs[t_order]- self.Wfe2* E2[t_order]

                    """
                    print('time:\t\t',t_step)
                    print('P, E1, F,\t\t', P[t_order],E1[t_order],F[t_order])
                    print('P, E2, F,\t\t', P[t_order],E2[t_order],F[t_order])
                    print('cs, us,\t\t', cs[t_order],us[t_order])
                    print('Wp, We1, We2, Wf\t\t',Wp[t_order],We1[t_order],We2[t_order],Wf[t_order],'\n')
                    """
                    if ( (us[t_order]- F[t_order])>=0):
                        LToF[t_order]=  (us[t_order]- F[t_order])
                        dWf= self.alpha_F* cs[t_order]* (us[t_order]- F[t_order])
                        
                        "print ('dwf', dWf)"
                    else:
                        dWf=0
                        

                    if ((us[t_order]-P[t_order]) >=0):

                        LToP[t_order]=  (us[t_order]- P[t_order])
                        dWp= self.alpha_P* cs[t_order]* (us[t_order]- P[t_order])
                        "print ('dwp', dWp)"

                    else:
                        dWp= 0
                    
                    if ((F[t_order]*(P[t_order]- us[t_order]- E1[t_order])) >=0 ):

                        LToE1[t_order]=  (F[t_order]*(P[t_order]- us[t_order]- E1[t_order]))
                        dWe1= self.alpha_E1* cs[t_order]* (F[t_order]*( P[t_order]-us[t_order]- E1[t_order]))
                        "print ('dwe', dWe)"
                    else:
                        dWe1=0

                    if ((F[t_order]*(P[t_order]- us[t_order]- E2[t_order])) >=0 ):

                        LToE2[t_order]= (F[t_order]*(P[t_order]- us[t_order]- E2[t_order]))
                        dWe2=   self.alpha_E2* cs[t_order]* (F[t_order]*( P[t_order]-us[t_order]- E2[t_order]))- self.beta_E2* (We2[t_order]-_We2[t_order])
                        "print ('dwe', dWe)"
                    else:
                        dWe2= 0

                    d_We2= self._alphaE2* cs[t_order]* E1[t_order]- self._betaE2* (_We2[t_order]- We2[t_order])

                  
                    
                else:
                    P[t_order]= P[t_order-1]        
                    E1[t_order]=    E1[t_order-1]        
                    E2[t_order]=    E2[t_order-1]        
                    F[t_order]= F[t_order-1]

                    Wf[t_order]= Wf[t_order-1]        
                    We1[t_order]= We1[t_order-1]        
                    We2[t_order]= We2[t_order-1]        
                    Wp[t_order]= Wp[t_order-1]
                    
                    
                    """
                    LToF[t_order]=  LToF[t_order-1] 
                    LToE[t_order]=  LToE[t_order-1] 
                    LToP[t_order]=  LToP[t_order-1] 
                    """
                    """
                    print('P, E1, F,\t', P[t_order],E1[t_order],F[t_order])
                    print('P, E2, F,\t', P[t_order],E2[t_order],F[t_order])
                    print('cs, us,\t\t', cs[t_order],us[t_order])
                    print('Wp, We1, We2, Wf',Wp[t_order],We1[t_order],We2[t_order],Wf[t_order],'\n')
                    """
                   
            if  (t_step <= 60) &  (t_step>30) :
                
                

                Wp[t_order] =   Wp[t_order-1]+ self.dt* dWp
                We1[t_order] =  We1[t_order-1]+ self.dt* dWe1
                We2[t_order] =  We2[t_order-1]+ self.dt* dWe2
                Wf[t_order] =   Wf[t_order-1]+ self.dt* dWf
                _We2[t_order] = _We2[t_order-1]+ self.dt* d_We2

                P[t_order]= Wp[t_order]* cs[t_order]
                E1[t_order]= We1[t_order]* cs[t_order] + vmpfc[t_order]
                E2[t_order]= We2[t_order]* cs[t_order]+ self.We2e1* E1[t_order] 
                F[t_order]= Wf[t_order]* cs[t_order]- self.Wfe2* E2[t_order]
                    
                 
                if ( (us[t_order]- F[t_order])>=0):
                    LToF[t_order]=  (us[t_order]- F[t_order])
                    dWf= self.alpha_F* cs[t_order]* (us[t_order]- F[t_order])
                        
                    "print ('dwf', dWf)"
                else:
                    dWf=0
                        

                if ((us[t_order]-P[t_order]) >=0):

                    LToP[t_order]=  (us[t_order]- P[t_order])
                    dWp= self.alpha_P* cs[t_order]* (us[t_order]- P[t_order])
                    "print ('dwp', dWp)"

                else:
                    dWp= 0
                    
                if ((F[t_order]*(P[t_order]- us[t_order]- E1[t_order])) >=0 ):

                    LToE1[t_order]=  (F[t_order]*(P[t_order]- us[t_order]- E1[t_order]))
                    dWe1= self.alpha_E1* cs[t_order]* (F[t_order]*( P[t_order]-us[t_order]- E1[t_order]))
                    "print ('dwe', dWe)"
                else:
                    dWe1=0

                if ((F[t_order]*(P[t_order]- us[t_order]- E2[t_order])) >=0 ):

                    LToE2[t_order]= (F[t_order]*(P[t_order]- us[t_order]- E2[t_order]))
                    dWe2=   self.alpha_E2* cs[t_order]* (F[t_order]*( P[t_order]-us[t_order]- E2[t_order]))- self.beta_E2* (We2[t_order]-_We2[t_order])
                    "print ('dwe', dWe)"
                else:
                    dWe2= 0

                d_We2= self._alphaE2* cs[t_order]* E1[t_order]- self._betaE2* (_We2[t_order]- We2[t_order])
            
            """
            if (t_step>=10) & (t_step<30):

                Wp[t_order] =   Wp[t_order-1]+ self.dt* dWp
                We1[t_order] =  We1[t_order-1]+ self.dt* dWe1
                We2[t_order] =  We2[t_order-1]+ self.dt* dWe2
                Wf[t_order] =   Wf[t_order-1]+ self.dt* dWf
                _We2[t_order] = _We2[t_order-1]+ self.dt* d_We2

                P[t_order]= Wp[t_order]* cs[t_order]
                E1[t_order]= 10
                E2[t_order]= We2[t_order]* cs[t_order]+ self.We2e1* E1[t_order] 
                F[t_order]= Wf[t_order]* cs[t_order]- self.Wfe2* E2[t_order]
                    
                 
                if ( (us[t_order]- F[t_order])>=0):
                    LToF[t_order]=  (us[t_order]- F[t_order])
                    dWf= self.alpha_F* cs[t_order]* (us[t_order]- F[t_order])
                        
                    "print ('dwf', dWf)"
                else:
                    dWf=0
                        

                if ((us[t_order]-P[t_order]) >=0):

                    LToP[t_order]=  (us[t_order]- P[t_order])
                    dWp= self.alpha_P* cs[t_order]* (us[t_order]- P[t_order])
                    "print ('dwp', dWp)"

                else:
                    dWp= 0
                    
                if ((F[t_order]*(P[t_order]- us[t_order]- E1[t_order])) >=0 ):

                    LToE1[t_order]=  (F[t_order]*(P[t_order]- us[t_order]- E1[t_order]))
                    dWe1= self.alpha_E1* cs[t_order]* (F[t_order]*( P[t_order]-us[t_order]- E1[t_order]))
                    "print ('dwe', dWe)"
                else:
                    dWe1=0

                if ((F[t_order]*(P[t_order]- us[t_order]- E2[t_order])) >=0 ):

                    LToE2[t_order]= (F[t_order]*(P[t_order]- us[t_order]- E2[t_order]))
                    dWe2=   self.alpha_E2* cs[t_order]* (F[t_order]*( P[t_order]-us[t_order]- E2[t_order]))- self.beta_E2* (We2[t_order]-_We2[t_order])
                    "print ('dwe', dWe)"
                else:
                    dWe2= 0

                d_We2= self._alphaE2* cs[t_order]* E1[t_order]- self._betaE2* (_We2[t_order]- We2[t_order])

            """
            print("E1, E2, P, F\t\t", E1[t_order],E2[t_order],P[t_order],F[t_order])    
        return self.t, F, E1, E2, P, cs, us, Wf, We1, We2, Wp, vmpfc





afig=plt.figure()
a=Basic_Neural_Circuit_()

"""t=np.arange(100)
t=sp.arange(0,40,.001) # time domain
cs_at=sp.arange(1,40,1)   # time rate of CU     
us_at=np.array([1, 2, 5 ,8 ,9, 12, 13, 14, 17, 19])   # time rate of CU     
shape=a.rect(.1)        # shape and duration of stimulus
plt.subplot(2,1,1)
plt.title('stimulus schedule')
plt.ylabel('CS signal')
cs=a.pulse_train(t,cs_at,shape)
plt.plot(t,cs)
plt.subplot(2,1,2)
plt.ylabel('US signal')
us=a.pulse_train(t,us_at,shape)
plt.plot(t,us)

for i,j in enumerate(t):
    print(j, cs[i])
"""

t, f, e, p, cs, us, WF, WE, WP, le, lf, lp= a._activity()

plt.subplot(5,1,1)
plt.ylabel('CS')
plt.title('P(US|CS) = 1')
plt.plot(t,cs)

plt.subplot(5,1,2)
plt.plot(t,us)
plt.ylabel('US')

plt.subplot(5,1,3)

plt.plot(t,f,'k',label='Fear')
plt.plot(t,e,'r',label='Extinction')
plt.plot(t,p,'b',label='Persistant')
plt.ylabel('Activity')
plt.legend()

plt.subplot(5,1,4)
plt.plot(t,WF,'k',label='Fear')
plt.plot(t,WE,'r',label='Extinction')
plt.plot(t,WP,'b',label='Persistant')
plt.ylabel('Weight')
plt.legend()

plt.subplot(5,1,5)
plt.plot(t,lf,'k',label='Fear')
plt.plot(t,le,'r',label='Extinction')
plt.plot(t,lp,'b',label='Persistant')
plt.ylabel('Learning Signal')
plt.legend()

"""
bfig=plt.figure()

b=Extended_Neural_Circuit_()
t, f, e1, e2, p, cs, us, WF, WE1, WE2, WP, VMPFC =b._activity()

plt.subplot(4,1,1)
plt.ylabel('CS')
plt.title('Activation of vmPFC durin extinction')
plt.plot(t,cs)

plt.subplot(4,1,2)
plt.plot(t,us)
plt.ylabel('US')

plt.subplot(4,1,3)
"""

"""
vt = np.linspace(10, 30, 500)
vmp=2.5*(signal.sawtooth(2 * np.pi * 5 * vt) + 1)
plt.plot(vt, vmp)
plt.plot(t,VMPFC,'c')
plt.ylabel('vmPFC Activation')

plt.subplot(4,1,4)
plt.plot(t,f,'k',label='CEA')
plt.plot(t,e1,'g',label='vmPFC')
plt.plot(t,e2,'r',label='ITC')
plt.plot(t,p,'b',label='LA')
plt.ylabel('Activity')
plt.legend()

plt.subplot(4,1,4)
plt.plot(t,WF,'k',label='CEA')
plt.plot(t,WE1,'g',label='vmPFC')
plt.plot(t,WE2,'r',label='ITC')
plt.plot(t,WP,'b',label='LA')
plt.ylabel('Weight')
plt.legend()
"""


plt.show()



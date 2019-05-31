from brian import *

# standard filter group model
standard_filtergroup_model_params = Parameters(
    Vr=-60*mV,
    Vt=-50*mV,
    El=-60*mV,
    R=0.2*ohm,
    tau=1*ms,
    dVt=0*mV,
    mu=0*mV, sigma=1*mV,
    )
def standard_filtergroup_model_init(G, params):
    G.v = params.Vr+(params.Vt-params.Vr)*rand(len(G))
standard_filtergroup_model = {
    'eqs':'''
          dv/dt = (-(v-El)+R*I)/tau + mu/tau + sigma*xi*(2/tau)**0.5 : volt
          I : amp
          target_var = I
          ''',
    'parameters':standard_filtergroup_model_params,
    'threshold':standard_filtergroup_model_params.Vt,
    'reset':standard_filtergroup_model_params.Vr,
    'refractory':5*ms,
    'init':standard_filtergroup_model_init,
    'compress':lambda x:x**(1.0/3.0),
    }

# standard coincidence detector model
standard_cd_model_params = Parameters(
    tau=1*ms,
    sigma=.1,
    )
def standard_cd_model_init(G, params):
    G.v = rand(len(G))
standard_cd_model = {
    'eqs':'''
          dv/dt = -v/tau+sigma*(2./tau)**.5*xi : 1
          target_var = v
          ''',
    'parameters':standard_cd_model_params,
    'threshold':1,
    'reset':0,
    'refractory':0*ms,
    'init':standard_cd_model_init,
    'weight':.5,
    }

def makemodel(model, N, clock):
    eqs = Equations(model['eqs'], **model['parameters'])
    group = NeuronGroup(N,
                        eqs,
                        threshold=model['threshold'],
                        reset=model['reset'],
                        refractory=model['refractory'],
                        clock=clock)
    initmodel(model, group)
    return group

def initmodel(model, group):
    model['init'](group, model['parameters'])

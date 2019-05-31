from shared import *
from hrtf_analysis import *
from models import *
import gc

class ApproximateFilteringModel(object):
    '''
    Initialise this object with an hrtfset, a cochlear range (cfmin, cfmax, cfN),
    and optionally:
    a model for the coincidence detector neurons (cd_model),
    a model for the filter neurons (filtergroup_model),
    whether or not to use the correlation-based best delays (use_delays),
    whether or not to use the best gains (use_gains),
    whether or not to use only the phase information (delays between -pi and pi),
    an alternative set of itd/ild pairs (itdild, see the file hrtf_analysis.py
    for more information on this, function hrtfset_itd_ild).
    
    The __call__ method returns a count (see docstring of that method). 
    '''
    def __init__(self, hrtfset, cfmin, cfmax, cfN,
                 cd_model=standard_cd_model,
                 filtergroup_model=standard_filtergroup_model,
                 use_delays=True, use_gains=True, use_only_phase=False,
                 itdild=None,
                 ):
        self.hrtfset = hrtfset
        self.cfmin, self.cfmax, self.cfN = cfmin, cfmax, cfN
        self.cd_model = cd_model
        self.filtergroup_model = filtergroup_model
        
        self.num_indices = num_indices = hrtfset.num_indices
        cf = erbspace(cfmin, cfmax, cfN)
        
        # extract ITDs/ILDs in the right form
        if itdild is None:
            all_itds, all_ilds = hrtfset_itd_ild(hrtfset, cfmin, cfmax, cfN)
        else:
            all_itds, all_ilds = itdild
        d = array([all_itds[j][i] for i in xrange(cfN) for j in xrange(num_indices)])
        g = array([all_ilds[j][i] for i in xrange(cfN) for j in xrange(num_indices)])
        gains = hstack((1/g, g))
        gains_dB = 20*log10(gains)
        abs_gains_dB = abs(gains_dB)
        r = -abs_gains_dB[:len(gains)/2]
        r = hstack((r, r))
        gains_dB += r
        gains = 10**(gains_dB/20)
        gains = reshape(gains, (1, len(gains)))

        if use_only_phase:
            d_cf = repeat(cf, num_indices)
            d = imag(log(exp(1j*2*pi*d*d_cf)))/(2*pi*d_cf) # delays constrained to having their phase be in [-pi, pi] 

        delays_L = where(d>=0, zeros(len(d)), -d)
        delays_R = where(d>=0, d, zeros(len(d)))
        delay_max = max(amax(delays_L), amax(delays_R))*second

        if not use_gains:
            gains = ones(len(gains))

        if not use_delays:
            delays_L = delays_R = zeros(len(d))
            delay_max = 2/samplerate
        
        # dummy sound, when we run apply() we replace it
        sound = Sound((silence(1*ms), silence(1*ms)))
        soundinput = DoNothingFilterbank(sound)
        
        gfb = Gammatone(Repeat(soundinput, cfN), hstack((cf, cf)))
        
        gains_fb = FunctionFilterbank(Repeat(gfb, num_indices),
                                      lambda x:x*gains)
        
        compress = filtergroup_model['compress']
        cochlea = FunctionFilterbank(gains_fb, lambda x:compress(clip(x, 0, Inf)))
        
        # Create the filterbank group
        eqs = Equations(filtergroup_model['eqs'], **filtergroup_model['parameters'])
        G = FilterbankGroup(cochlea, 'target_var', eqs,
                            threshold=filtergroup_model['threshold'],
                            reset=filtergroup_model['reset'],
                            refractory=filtergroup_model['refractory'])
        
        # create the synchrony group
        cd_eqs = Equations(cd_model['eqs'], **cd_model['parameters'])
        cd = NeuronGroup(num_indices*cfN, cd_eqs,
                         threshold=cd_model['threshold'],
                         reset=cd_model['reset'],
                         refractory=cd_model['refractory'],
                         clock=G.clock)
        
        # set up the synaptic connectivity
        cd_weight = cd_model['weight']
        C = Connection(G, cd, 'target_var', delay=True, max_delay=delay_max)
        for i in xrange(num_indices*cfN):
            C[i, i] = cd_weight
            C[i+num_indices*cfN, i] = cd_weight
            C.delay[i, i] = delays_L[i]
            C.delay[i+cfN*num_indices, i] = delays_R[i]

        self.soundinput = soundinput
        self.filtergroup = G
        self.synchronygroup = cd
        self.synapses = C
        self.counter = SpikeCounter(cd)
        self.network = Network(G, cd, C, self.counter)
        
    def __call__(self, sound, index=None, **indexkwds):
        '''
        Apply approximate filtering group to given sound, which should be a
        stereo sound unless you specify the HRTF index, or coordinates of
        the HRTF index as keyword arguments, in which case it should be a mono
        sound which will have the given HRTF applied to it. You can also
        specify index=hrtf. Returns the spike count of the neurons in the synchrony
        group with shape (cfN, num_indices).
        '''
        hrtf = None
        if index is not None:
            hrtf = self.hrtfset[index]
        elif isinstance(index, HRTF):
            hrtf = index
        elif len(indexkwds):
            hrtf = self.hrtfset(**indexkwds)
        if hrtf is not None:
            sound = hrtf(sound)
        self.soundinput.source = sound
        self.network.reinit()
        self.filtergroup_model['init'](self.filtergroup,
                                       self.filtergroup_model['parameters'])
        self.cd_model['init'](self.synchronygroup, self.cd_model['parameters'])
        self.network.run(sound.duration, report='stderr')
        count = reshape(self.counter.count, (self.cfN, self.num_indices))
        return count

if __name__=='__main__':
    
    from plot_count import ircam_plot_count

    hrtfdb = get_ircam()
    subject = 1002
    hrtfset = hrtfdb.load_subject(subject)
    index = randint(hrtfset.num_indices)
    cfmin, cfmax, cfN = 150*Hz, 5*kHz, 80
    sound = whitenoise(500*ms)
    
    afmodel = ApproximateFilteringModel(hrtfset, cfmin, cfmax, cfN)
    
    count = afmodel(sound, index)
    
    ircam_plot_count(hrtfset, count, index=index)
    show()

from shared import *
from hrtf_analysis import *
from models import *
import gc

class IdealFilteringModel(object):
    '''
    Initialise this object with an hrtfset, a cochlear range (cfmin, cfmax, cfN),
    and optionally:
    a model for the coincidence detector neurons (cd_model),
    a model for the filter neurons (filtergroup_model),
    whether or not to normalise the cochlear-filtered HRTFs, which improves
    performance by making each frequency band have the same power (and therefore
    comparable firing rates in the neurons) (use_normalisation_gains).
    
    The __call__ method returns a count (see docstring of that method). 
    '''
    def __init__(self, hrtfset, cfmin, cfmax, cfN,
                 cd_model=standard_cd_model,
                 filtergroup_model=standard_filtergroup_model,
                 use_normalisation_gains=True,
                 ):
        self.hrtfset = hrtfset
        self.cfmin, self.cfmax, self.cfN = cfmin, cfmax, cfN
        self.cd_model = cd_model
        self.filtergroup_model = filtergroup_model
        
        self.num_indices = num_indices = hrtfset.num_indices
        cf = erbspace(cfmin, cfmax, cfN)
                
        # dummy sound, when we run apply() we replace it
        sound = Sound((silence(1*ms), silence(1*ms)))
        soundinput = DoNothingFilterbank(sound)
        
        hrtfset_fb = hrtfset.filterbank(
                RestructureFilterbank(soundinput, 
                        indexmapping=repeat([1, 0], hrtfset.num_indices)))

        # We normalise the different HRTFs because we don't want a stronger
        # response from channels with less attenuation in the HRTF, but rather
        # a stronger response when the filters are more closely equal
        if use_normalisation_gains:
            attenuations = hrtfset_attenuations(cfmin, cfmax, cfN, hrtfset)
            #shape: (2, hrtfset.num_indices, cfN))
            gains_max = reshape(1/maximum(attenuations[0], attenuations[1]), (1, hrtfset.num_indices, cfN))
            gains = vstack((gains_max, gains_max))
            gains.shape = gains.size
            func = lambda x: x*gains
        else:
            func = lambda x: x

        gains_fb = FunctionFilterbank(Repeat(hrtfset_fb, cfN), func)

        gfb = Gammatone(gains_fb,
                        tile(cf, hrtfset_fb.nchannels))
        
        compress = filtergroup_model['compress']
        cochlea = FunctionFilterbank(gfb, lambda x:compress(clip(x, 0, Inf)))
        
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
        C = Connection(G, cd, 'target_var')
        for i in xrange(num_indices*cfN):
            C[i, i] = cd_weight
            C[i+num_indices*cfN, i] = cd_weight

        self.soundinput = soundinput
        self.filtergroup = G
        self.synchronygroup = cd
        self.synapses = C
        self.counter = SpikeCounter(cd)
        self.network = Network(G, cd, C, self.counter)
        
    def __call__(self, sound, index=None, **indexkwds):
        '''
        Apply ideal filtering group to given sound, which should be a
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
        count = reshape(self.counter.count, (self.num_indices, self.cfN)).T
        return count

if __name__=='__main__':
    
    from plot_count import ircam_plot_count

    hrtfdb = get_ircam()
    subject = 1002
    hrtfset = hrtfdb.load_subject(subject)
    index = randint(hrtfset.num_indices)
    cfmin, cfmax, cfN = 150*Hz, 5*kHz, 80
    sound = whitenoise(500*ms)
    
    ifmodel = IdealFilteringModel(hrtfset, cfmin, cfmax, cfN)
    
    count = ifmodel(sound, index)
    
    ircam_plot_count(hrtfset, count, index=index)
    show()

from shared import *
from hrtf_analysis import *
from models import *
import gc

class AllPairsModel(object):
    '''
    Initialise this object with an hrtfset, a cochlear range (cfmin, cfmax, cfN),
    a range of gains (gain_max in dB, gain_N) and a range of delays (delay_max,
    delay_N),
    and optionally:
    a model for the coincidence detector neurons (cd_model),
    a model for the filter neurons (filtergroup_model).
        
    The __call__ method returns a count (see docstring of that method). 
    '''
    def __init__(self, hrtfset, cfmin, cfmax, cfN,
                 gain_max, gain_N, delay_max, delay_N,
                 cd_model=standard_cd_model,
                 filtergroup_model=standard_filtergroup_model,
                 ):
        self.hrtfset = hrtfset
        self.cfmin, self.cfmax, self.cfN = cfmin, cfmax, cfN
        self.cd_model = cd_model
        self.filtergroup_model = filtergroup_model
        self.gain_max = gain_max
        self.gain_N = gain_N
        self.delay_max = delay_max
        self.delay_N = delay_N
        
        self.num_indices = num_indices = hrtfset.num_indices
        cf = erbspace(cfmin, cfmax, cfN)
                
        # dummy sound, when we run apply() we replace it
        sound = Sound((silence(1*ms), silence(1*ms)))
        soundinput = DoNothingFilterbank(sound)

        # prepare gains filter
        m = (gain_N+1)/2
        gains_dB = linspace(0, gain_max, m)
        gains = 10**(gains_dB/20)
        gains = hstack((1/gains[::-1], gains[1:]))
        allgains = reshape(gains, (1, 1, gains.size))

        def apply_gains(y):
            nsamples = y.shape[0]
            cfN = y.shape[1]/2
            y = reshape(y, (nsamples, 2*cfN, 1))            
            y1 = y[:, :cfN, :]*allgains
            y2 = y[:, cfN:, :]*allgains[:, :, ::-1]
            y = hstack((y1, y2))
            y = reshape(y, (nsamples, y.size/nsamples))
            return y
        
        gfb = Gammatone(Repeat(soundinput, cfN), hstack((cf, cf)))
                
        gains_fb = FunctionFilterbank(gfb, apply_gains)
        gains_fb.nchannels = gfb.nchannels*gain_N
        
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
        cd = NeuronGroup(cfN*gain_N*(delay_N*2-1), cd_eqs,
                         threshold=cd_model['threshold'],
                         reset=cd_model['reset'],
                         refractory=cd_model['refractory'],
                         clock=G.clock)
        
        # set up the synaptic connectivity
        left_delays = hstack((zeros(delay_N-1), linspace(0, float(delay_max), delay_N)))
        right_delays = left_delays[::-1]
        cd_weight = cd_model['weight']
        C = Connection(G, cd, 'target_var', delay=True, max_delay=delay_max)
        for i, j, dl, dr in zip(repeat(arange(cfN*gain_N), 2*delay_N-1),
                                arange(cfN*gain_N*(delay_N*2-1)),
                                tile(left_delays, cfN*gain_N),
                                tile(right_delays, cfN*gain_N)):
            C[i, j] = cd_weight
            C[i+cfN*gain_N, j] = cd_weight
            C.delay[i, j] = dl
            C.delay[i+cfN*gain_N, j] = dr

        self.soundinput = soundinput
        self.filtergroup = G
        self.synchronygroup = cd
        self.synapses = C
        self.counter = SpikeCounter(cd)
        self.network = Network(G, cd, C, self.counter)
        
    def __call__(self, sound, index=None, **indexkwds):
        '''
        Apply all pairs filtering group to given sound, which should be a
        stereo sound unless you specify the HRTF index, or coordinates of
        the HRTF index as keyword arguments, in which case it should be a mono
        sound which will have the given HRTF applied to it. You can also
        specify index=hrtf. Returns the count of the neurons in the synchrony
        group with shape (cfN, gain_N, delay_N*2-1).
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
        count = reshape(self.counter.count,
                        (self.cfN, self.gain_N, self.delay_N*2-1))
        return count

if __name__=='__main__':
    
    from plot_count import ircam_plot_count

    hrtfdb = get_ircam()
    subject = 1002
    hrtfset = hrtfdb.load_subject(subject)
    index = randint(hrtfset.num_indices)
    cfmin, cfmax, cfN = 150*Hz, 5*kHz, 80
    gain_max, gain_N = 8.0, 61
    delay_N = 35
    delay_max = delay_N/samplerate
    # Change this to 10*second for equivalent picture to the paper
    sound = whitenoise(200*ms).atlevel(80*dB)
    
    apmodel = AllPairsModel(hrtfset, cfmin, cfmax, cfN,
                            gain_max, gain_N, delay_max, delay_N)
    
    count = apmodel(sound, index)

    # Complicated code to plot the output nicely
    freqlabels = array([150*Hz, 1*kHz, 2*kHz, 3*kHz, 4*kHz, 5*kHz])
    fig_mew = 1 # marker edge width (in points)
    num_indices = hrtfset.num_indices
    from scipy.ndimage.filters import *
    itd, ild = hrtfset_itd_ild(hrtfset, cfmin, cfmax, cfN)
    delays = array([itd[index][i] for i in xrange(cfN)])
    gains = array([ild[index][i] for i in xrange(cfN)])
    gains = 20*log10(gains)
    delays = -array(delays*samplerate, dtype=int)+delay_N-1
    arrgains = linspace(-gain_max, gain_max, gain_N)
    gains = digitize(gains, 0.5*(arrgains[1:]+arrgains[:-1]))
    gains = gain_N-1-gains
    def dofig(count, blur=0, blurmode='reflect', freqlabels=None):
        count = array(count, dtype=float) 
        ocount = count
        count = copy(ocount)
        count.shape = (cfN, gain_N, delay_N*2-1)
        count = amax(count, axis=1)
        count.shape = (cfN, delay_N*2-1)
        subplot(121)
        count = gaussian_filter(count, blur, mode=blurmode)
        imshow(count, origin='lower left', interpolation='nearest', aspect='auto',
               extent=(-float(delay_N/samplerate/msecond), float(delay_N/samplerate/msecond), 0, cfN))
        plot((delays-delay_N)/samplerate/msecond, arange(cfN), '+', color=(0,0,0), mew=fig_mew)
        plot((argmax(count, axis=1)-delay_N)/samplerate/msecond, arange(cfN), 'x', color=(1,1,1), mew=fig_mew)
        axis((float(-delay_N/samplerate/msecond), float(delay_N/samplerate/msecond), 0, cfN))
        xlabel('Delay (ms)')
        if freqlabels is None:
            yticks([])
            ylabel('Channel')
        else:
            cf = erbspace(cfmin, cfmax, cfN)
            j = digitize(freqlabels, .5*(cf[1:]+cf[:-1]))
            yticks(j, map(str, array(freqlabels, dtype=int)))
            ylabel('Channel (Hz)')
        subplot(122)
        count = copy(ocount)
        count.shape = (cfN, gain_N, delay_N*2-1)
        count = amax(count, axis=2)
        count.shape = (cfN, gain_N)
        count = gaussian_filter(count, blur, mode=blurmode)
        imshow(count, origin='lower left', interpolation='nearest', aspect='auto')
        plot(gains, arange(cfN), '+', color=(0,0,0), mew=fig_mew)
        plot(argmax(count, axis=1), arange(cfN), 'x', color=(1,1,1), mew=fig_mew)
        axis('tight')
        xlabel('Relative gain (dB)')
        xticks([0, (gain_N-1)/2, gain_N-1], [str(min(arrgains)), '0', str(max(arrgains))])
        if freqlabels is None:
            yticks([])
            ylabel('Channel')
        else:
            cf = erbspace(cfmin, cfmax, cfN)
            j = digitize(freqlabels, .5*(cf[1:]+cf[:-1]))
            yticks(j, map(str, array(freqlabels, dtype=int)))
            ylabel('Channel (Hz)')
    dofig(count, freqlabels=freqlabels)
    figure()
    dofig(count, blur=1)#, freqlabels=[500, 1000, 2000, 3000, 4000, 5000])
    figure()
    dofig(count, blur=2)
    show()

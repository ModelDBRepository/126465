from shared import *

def hrtfset_itd_ild(hrtfset, cfmin, cfmax, cfN):
    cf = erbspace(cfmin, cfmax, cfN)
    man_name = hrtfset.name+'-'+str((int(cfmin)))+'-'+str(int(cfmax))+'-'+str(cfN)
    fname = datapath+'/itdild/'+man_name+'.pkl'
    if os.path.exists(fname):
        return pickle.load(open(fname, 'rb'))
    all_itds = []
    all_ilds = []
    num_indices = hrtfset.num_indices
    for j in xrange(num_indices):
        hrir = Sound(hrtfset.hrtf[j].fir.T, samplerate=hrtfset.samplerate)
        fb = Gammatone(Repeat(hrir, cfN), hstack((cf, cf)))
        filtered_hrirset = fb.process()
        itds = []
        ilds = []
        for i in xrange(cfN):
            left = filtered_hrirset[:, i]
            right = filtered_hrirset[:, i+cfN]
            # This FFT stuff does a correlate(left, right, 'full')
            Lf = fft(hstack((left, zeros(len(left)))))
            Rf = fft(hstack((right[::-1], zeros(len(right)))))
            C = ifft(Lf*Rf).real
            i = argmax(C)+1-len(left)
            itds.append(i/hrtfset.samplerate)
            ilds.append(sqrt(amax(C)/sum(right**2)))
        itds = array(itds)
        ilds = array(ilds)
        all_itds.append(itds)
        all_ilds.append(ilds)
    pickle.dump((all_itds, all_ilds), open(fname, 'wb'), -1)
    return all_itds, all_ilds

def hrtfset_attenuations(cfmin, cfmax, cfN, hrtfset, sound=None):
    fname = datapath+'/hrtf_attenuation/'+hrtfset.name+'-'+str((int(cfmin)))+'-'+str(int(cfmax))+'-'+str(cfN)+'.pkl'
    if os.path.exists(fname):
        return pickle.load(open(fname, 'rb'))
    
    sound = Sound(array([1.]))[:40*ms]
    cf = erbspace(cfmin, cfmax, cfN)

    hrtfset_fb = hrtfset.filterbank(Repeat(sound, 2*hrtfset.num_indices))
    gfb = Gammatone(Repeat(hrtfset_fb, cfN),
                    tile(cf, hrtfset_fb.nchannels))

    gfb2 = Gammatone(sound, cf)
    
    output_fb = Join(gfb, gfb2)

    y = zeros(2*hrtfset.num_indices*cfN)
    z = zeros(cfN)
    endpoints = hstack((arange(0, sound.shape[0], 32), sound.shape[0]))
    for start, end in zip(endpoints[:-1], endpoints[1:]):
        output = output_fb.buffer_fetch(start, end)
        output1 = output[:, :-cfN]
        output2 = output[:, -cfN:]
        y = maximum(y, amax(output1, axis=0))
        z = maximum(z, amax(output2, axis=0))
    y.shape = (2, hrtfset.num_indices, cfN)
    z.shape = (1, 1, cfN)
    y = y[::-1, :, :]
    y /= z
    
    pickle.dump(y, open(fname, 'wb'), -1)
    return y

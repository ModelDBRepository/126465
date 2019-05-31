from brian import *

def ircam_plot_count(hrtfset, count, index=None, showbest=True, absolute=False,
                     vmin=None, vmax=None, I=None, ms=20, mew=2, indexcol='k', bestcol='w'):
    count = array(count, dtype=float)
    num_indices = hrtfset.num_indices
    count = sum(count, axis=0)
    if I is None: I = arange(num_indices)
    img = zeros((10, 24))
    for i, c in enumerate(count):
        if i in I:
            elev = hrtfset.coordinates['elev'][i]
            azim = hrtfset.coordinates['azim'][i]
            if elev<60:
                w = 1
            elif elev==60:
                w = 2
            elif elev==75:
                w = 4
            elif elev==90:
                w = 24
                azim = -180
            if azim>=180: azim -= 360
            x = int((azim+180)/15)
            y = int((elev+45)/15)
            img[y, x:x+w] = c
    if absolute:
        imshow(img, origin='lower left', interpolation='nearest', extent=(-180-7.5, 180-7.5, -45-7.5, 90+7.5),
               vmin=vmin, vmax=vmax)
        axis('tight')
    else:
        imshow(img, origin='lower left', interpolation='nearest', extent=(-180-7.5, 180-7.5, -45-7.5, 90+7.5))
        axis('tight')
    if index is not None:
        azim = hrtfset.coordinates['azim'][index]
        elev = hrtfset.coordinates['elev'][index]
        if azim>=180: azim -= 360
        plot([azim], [elev], '+', ms=ms, mew=mew, color=indexcol)
    if showbest:
        i = argmax(count)
        azim = hrtfset.coordinates['azim'][i]
        elev = hrtfset.coordinates['elev'][i]
        if azim>=180: azim -= 360
        plot([azim], [elev], 'x', ms=ms, mew=mew, color=bestcol)
    return img

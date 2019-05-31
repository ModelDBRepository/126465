This code implements the models from:

	Goodman DFM, Brette R, 2010 Spike-Timing-Based Computation in Sound
	Localization. PLoS Comput Biol 6(11): e1000993.
	doi:10.1371/journal.pcbi.1000993

Code is not provided to generate all the data and the figures, but the three
basic models (approximate/ideal/allpairs) are shown.

Installation
------------

You will need to download and extract the IRCAM LISTEN database:

	http://recherche.ircam.fr/equipes/salles/listen/
	
The files should be extracted to a folder something like:

	F:\HRTF\IRCAM\IRC_1002
	etc.
	
You will need to change the file shared.py to give the location of this database
(see below).

In addition, you will need a copy of Python 2.5, 2.6 or 2.7 and the packages
numpy, scipy, matplotlib. You will also need the Brian neural network simulator
package, version 1.3 or above:

	http://www.briansimulator.org/

Guide to files
--------------
	
approximate_filtering_model.py
ideal_filtering_model.py

	The approximate and ideal filtering models.
	
all_pairs_model.py

	The learning model. The implementation of the model is included, but not
	code for generating the learned ITD/ILD pairs: this code is mostly just
        technical file management stuff, so it is not included for simplicity.
	
hrtf_analysis.py

	Generate best gain/delay pairs for the approximate filtering model, and
	find the normalisation factors for the ideal filtering model. Results are
	saved so only need to be generated once.
	
models.py

	The neural models used. Changing these equations and parameters can be used
	to easily switch to different models. Only the leaky integrate-and-fire
	model is given.
	
plot_count.py

	A function for plotting the outputs of the approximate/ideal filtering
	model, specialised for the IRCAM LISTEN database.
	
shared.py

	Various imports and variables that are shared across all of the models.
	You should change the ircam_locations variable in the get_ircam() function
	to reflect the location where you have saved the IRCAM data.

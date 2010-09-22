import os                                    # system functions

import nipype.algorithms.modelgen as model   # model generation
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.io as nio           # i/o routines
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine

from nipype.externals.pynifti import load

"""
Preliminaries
-------------

Confirm package dependencies are installed.  (This is only for the tutorial,
rarely would you put this in your own code.)
"""

from nipype.utils.misc import package_check

package_check('numpy', '1.3', 'tutorial1')
package_check('scipy', '0.7', 'tutorial1')
package_check('networkx', '1.0', 'tutorial1')
package_check('IPython', '0.10', 'tutorial1')

"""
Setup any package specific configuration. The output file format for FSL
routines is being set to compressed NIFTI.
"""

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

"""
Setting up workflows
--------------------

In this tutorial we will be setting up a hierarchical workflow for fsl
analysis. This will demonstrate how pre-defined workflows can be setup and
shared across users, projects and labs.


Setup preprocessing workflow
----------------------------

This is a generic fsl feat preprocessing workflow encompassing skull stripping,
motion correction and smoothing operations.

"""

fslfspreproc = pe.Workflow(name='fslfspreproc')

"""
Set up a node to define all inputs required for the preprocessing workflow
"""

inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                             'fssubject_id',
                                                             'surf_dir']),
                    name='inputspec')

"""
Convert functional images to float representation. Since there can be more than
one functional run we use a MapNode to convert each run.
"""

img2float = pe.MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                             op_string = '',
                                             suffix='_dtype'),
                       iterfield=['in_file'],
                       name='img2float')
fslfspreproc.connect(inputnode, 'func', img2float, 'in_file')

"""
Extract the middle volume of the first run as the reference
"""

extract_ref = pe.Node(interface=fsl.ExtractROI(t_size=1,
                                               t_min=0),
                      name = 'extractref')

"""
Define a function to pick the first file from a list of files
"""

def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files

fslfspreproc.connect(img2float, ('out_file', pickfirst), extract_ref, 'in_file')

"""
Realign the functional runs to the middle volume of the first run
"""

motion_correct = pe.MapNode(interface=fsl.MCFLIRT(save_mats = True,
                                                  save_plots = True),
                            name='realign',
                            iterfield = ['in_file'])
fslfspreproc.connect(img2float, 'out_file', motion_correct, 'in_file')
fslfspreproc.connect(extract_ref, 'roi_file', motion_correct, 'ref_file')

"""
Plot the estimated motion parameters
"""

plot_motion = pe.MapNode(interface=fsl.PlotMotionParams(in_source='fsl'),
                        name='plot_motion',
                        iterfield=['in_file'])
plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
fslfspreproc.connect(motion_correct, 'par_file', plot_motion, 'in_file')

"""
Extract the mean volume of the first functional run
"""

meanfunc = pe.Node(interface=fsl.ImageMaths(op_string = '-Tmean',
                                            suffix='_mean'),
                   name='meanfunc')
fslfspreproc.connect(motion_correct, ('out_file', pickfirst), meanfunc, 'in_file')

"""
Strip the skull from the mean functional to generate a mask
"""

meanfuncmask = pe.Node(interface=fsl.BET(mask = True,
                                         no_output=True,
                                         frac = 0.3),
                       name = 'meanfuncmask')
fslfspreproc.connect(meanfunc, 'out_file', meanfuncmask, 'in_file')

"""
Mask the functional runs with the extracted mask
"""

maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_bet',
                                               op_string='-mas'),
                      iterfield=['in_file'],
                      name = 'maskfunc')
fslfspreproc.connect(motion_correct, 'out_file', maskfunc, 'in_file')
fslfspreproc.connect(meanfuncmask, 'mask_file', maskfunc, 'in_file2')


"""
Determine the 2nd and 98th percentile intensities of each functional run
"""

getthresh = pe.MapNode(interface=fsl.ImageStats(op_string='-p 2 -p 98'),
                       iterfield = ['in_file'],
                       name='getthreshold')
fslfspreproc.connect(maskfunc, 'out_file', getthresh, 'in_file')


"""
Threshold the first run of the functional data at 10% of the 98th percentile
"""

threshold = pe.Node(interface=fsl.ImageMaths(out_data_type='char',
                                             suffix='_thresh'),
                       name='threshold')
fslfspreproc.connect(maskfunc, ('out_file', pickfirst), threshold, 'in_file')

"""
Define a function to get 10% of the intensity
"""

def getthreshop(thresh):
    return '-thr %.10f -Tmin -bin'%(0.1*thresh[0][1])
fslfspreproc.connect(getthresh, ('out_stat', getthreshop), threshold, 'op_string')

"""
Determine the median value of the functional runs using the mask
"""

medianval = pe.MapNode(interface=fsl.ImageStats(op_string='-k %s -p 50'),
                       iterfield = ['in_file'],
                       name='medianval')
fslfspreproc.connect(motion_correct, 'out_file', medianval, 'in_file')
fslfspreproc.connect(threshold, 'out_file', medianval, 'mask_file')

"""
Dilate the mask
"""

dilatemask = pe.Node(interface=fsl.ImageMaths(suffix='_dil',
                                              op_string='-dilF'),
                       name='dilatemask')
fslfspreproc.connect(threshold, 'out_file', dilatemask, 'in_file')

"""
Mask the motion corrected functional runs with the dilated mask
"""

maskfunc2 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                op_string='-mas'),
                      iterfield=['in_file'],
                      name='maskfunc2')
fslfspreproc.connect(motion_correct, 'out_file', maskfunc2, 'in_file')
fslfspreproc.connect(dilatemask, 'out_file', maskfunc2, 'in_file2')

"""
Determine the mean image from each functional run
"""

meanfunc2 = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       iterfield=['in_file'],
                       name='meanfunc2')
fslfspreproc.connect(maskfunc2, 'out_file', meanfunc2, 'in_file')

"""
Merge the median values with the mean functional images into a coupled list
"""

mergenode = pe.Node(interface=util.Merge(2, axis='hstack'),
                    name='merge')
fslfspreproc.connect(meanfunc2,'out_file', mergenode, 'in1')
fslfspreproc.connect(medianval,'out_stat', mergenode, 'in2')


"""
Identitynode to set fwhm
"""
smoothval = pe.Node(interface=util.IdentityInterface(fields=['fwhm']),
                    name='smoothval')
                    
                       
"""
Smooth each run using SUSAN with the brightness threshold set to 75% of the
median value for each run and a mask consituting the mean functional
"""

smooth = pe.MapNode(interface=fsl.SUSAN(),
                    iterfield=['in_file', 'brightness_threshold','usans'],
                    name='smooth')

"""
Define a function to get the brightness threshold for SUSAN
"""

def getbtthresh(medianvals):
    return [0.75*val for val in medianvals]
fslfspreproc.connect(smoothval, 'fwhm', smooth, 'fwhm')
fslfspreproc.connect(maskfunc2, 'out_file', smooth, 'in_file')
fslfspreproc.connect(medianval, ('out_stat', getbtthresh), smooth, 'brightness_threshold')
fslfspreproc.connect(mergenode, ('out', lambda x: [[tuple([val[0],0.75*val[1]])] for val in x]), smooth, 'usans')

"""
Mask the smoothed data with the dilated mask
"""

maskfunc3 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                op_string='-mas'),
                      iterfield=['in_file'],
                      name='maskfunc3')
fslfspreproc.connect(smooth, 'smoothed_file', maskfunc3, 'in_file')
fslfspreproc.connect(dilatemask, 'out_file', maskfunc3, 'in_file2')


concatnode = pe.Node(interface=util.Merge(2),
                     name='concat')
fslfspreproc.connect(maskfunc2,('out_file', lambda x:[x]), concatnode, 'in1')
fslfspreproc.connect(maskfunc3,('out_file', lambda x:[x]), concatnode, 'in2')

selectnode = pe.Node(interface=util.Select(),name='select')

fslfspreproc.connect(concatnode, 'out', selectnode, 'inlist')

def chooseindex(fwhm):
    if fwhm<1:
        return [0]
    else:
        return [1]
    
fslfspreproc.connect(smoothval, ('fwhm', chooseindex), selectnode, 'index')


"""
Scale the median value of the run is set to 10000
"""

meanscale = pe.MapNode(interface=fsl.ImageMaths(suffix='_gms'),
                      iterfield=['in_file','op_string'],
                      name='meanscale')
fslfspreproc.connect(selectnode, 'out', meanscale, 'in_file')

"""
Define a function to get the scaling factor for intensity normalization
"""

def getmeanscale(medianvals):
    return ['-mul %.10f'%(10000./val) for val in medianvals]
fslfspreproc.connect(medianval, ('out_stat', getmeanscale), meanscale, 'op_string')

"""
Perform temporal highpass filtering on the data
"""

highpass = pe.MapNode(interface=fsl.ImageMaths(suffix='_tempfilt'),
                      iterfield=['in_file'],
                      name='highpass')
fslfspreproc.connect(meanscale, 'out_file', highpass, 'in_file')

"""
Generate a mean functional image from the first run
"""

meanfunc3 = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       iterfield=['in_file'],
                      name='meanfunc3')
fslfspreproc.connect(highpass, ('out_file', pickfirst), meanfunc3, 'in_file')


"""
Register the mean functional to the freesurfer surface
"""

surfregister = pe.Node(interface=fs.BBRegister(init='fsl',
                                               contrast_type='t2'),
                     name = 'surfregister')


"""
Use :class:`nipype.algorithms.rapidart` to determine which of the
images in the functional series are outliers based on deviations in
intensity and/or movement.
"""

art = pe.Node(interface=ra.ArtifactDetect(use_differences = [True, False],
                                          use_norm = True,
                                          zintensity_threshold = 3,
                                          parameter_source = 'FSL',
                                          mask_type = 'file'),
              name="art")

# Get information from the FreeSurfer directories (brainmask, etc)
FreeSurferSource = pe.Node(interface=nio.FreeSurferSource(), name='fssource')

# Allow inversion of brainmask.mgz to volume (functional) space for alignment
ApplyVolTransform = pe.Node(interface=fs.ApplyVolTransform(),
                            name='warpbrainmask')
ApplyVolTransform.inputs.inverse = True 


convert2nii = pe.Node(interface=fs.MRIConvert(out_type='niigz'),name='convert2nii')


fslfspreproc.connect([(inputnode, surfregister,[('fssubject_id','subject_id'),
                                           ('surf_dir','subjects_dir')]),
                 (meanfunc2, surfregister,[(('out_file',pickfirst),'source_file')]),
                 (motion_correct, art, [('par_file','realignment_parameters')]),
                 (maskfunc2, art, [('out_file','realigned_files')]),
                 (dilatemask, art, [('out_file', 'mask_file')]),
                 (inputnode, FreeSurferSource,[('fssubject_id','subject_id')]),
                 (FreeSurferSource, ApplyVolTransform,[('brainmask','target_file')]),
                 (surfregister, ApplyVolTransform,[('out_reg_file','reg_file')]),
                 (meanfunc2, ApplyVolTransform,[(('out_file', pickfirst), 'source_file')]),
                 (ApplyVolTransform, convert2nii,[('transformed_file','in_file')])
                 ])


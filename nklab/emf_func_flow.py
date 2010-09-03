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

preproc = pe.Workflow(name='preproc')

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
preproc.connect(inputnode, 'func', img2float, 'in_file')

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

preproc.connect(img2float, ('out_file', pickfirst), extract_ref, 'in_file')

"""
Realign the functional runs to the middle volume of the first run
"""

motion_correct = pe.MapNode(interface=fsl.MCFLIRT(save_mats = True,
                                                  save_plots = True),
                            name='realign',
                            iterfield = ['in_file'])
preproc.connect(img2float, 'out_file', motion_correct, 'in_file')
preproc.connect(extract_ref, 'roi_file', motion_correct, 'ref_file')

"""
Extract the mean volume of the first functional run
"""

meanfunc = pe.Node(interface=fsl.ImageMaths(op_string = '-Tmean',
                                            suffix='_mean'),
                   name='meanfunc')
preproc.connect(motion_correct, ('out_file', pickfirst), meanfunc, 'in_file')

"""
Strip the skull from the mean functional to generate a mask
"""

meanfuncmask = pe.Node(interface=fsl.BET(mask = True,
                                         no_output=True,
                                         frac = 0.3),
                       name = 'meanfuncmask')
preproc.connect(meanfunc, 'out_file', meanfuncmask, 'in_file')

"""
Mask the functional runs with the extracted mask
"""

maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_bet',
                                               op_string='-mas'),
                      iterfield=['in_file'],
                      name = 'maskfunc')
preproc.connect(motion_correct, 'out_file', maskfunc, 'in_file')
preproc.connect(meanfuncmask, 'mask_file', maskfunc, 'in_file2')


"""
Determine the 2nd and 98th percentile intensities of each functional run
"""

getthresh = pe.MapNode(interface=fsl.ImageStats(op_string='-p 2 -p 98'),
                       iterfield = ['in_file'],
                       name='getthreshold')
preproc.connect(maskfunc, 'out_file', getthresh, 'in_file')


"""
Threshold the first run of the functional data at 10% of the 98th percentile
"""

threshold = pe.Node(interface=fsl.ImageMaths(out_data_type='char',
                                             suffix='_thresh'),
                       name='threshold')
preproc.connect(maskfunc, ('out_file', pickfirst), threshold, 'in_file')

"""
Define a function to get 10% of the intensity
"""

def getthreshop(thresh):
    return '-thr %.10f -Tmin -bin'%(0.1*thresh[0][1])
preproc.connect(getthresh, ('out_stat', getthreshop), threshold, 'op_string')

"""
Determine the median value of the functional runs using the mask
"""

medianval = pe.MapNode(interface=fsl.ImageStats(op_string='-k %s -p 50'),
                       iterfield = ['in_file'],
                       name='medianval')
preproc.connect(motion_correct, 'out_file', medianval, 'in_file')
preproc.connect(threshold, 'out_file', medianval, 'mask_file')

"""
Dilate the mask
"""

dilatemask = pe.Node(interface=fsl.ImageMaths(suffix='_dil',
                                              op_string='-dilF'),
                       name='dilatemask')
preproc.connect(threshold, 'out_file', dilatemask, 'in_file')

"""
Mask the motion corrected functional runs with the dilated mask
"""

maskfunc2 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                op_string='-mas'),
                      iterfield=['in_file'],
                      name='maskfunc2')
preproc.connect(motion_correct, 'out_file', maskfunc2, 'in_file')
preproc.connect(dilatemask, 'out_file', maskfunc2, 'in_file2')

"""
Determine the mean image from each functional run
"""

meanfunc2 = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       iterfield=['in_file'],
                       name='meanfunc2')
preproc.connect(maskfunc2, 'out_file', meanfunc2, 'in_file')

"""
Merge the median values with the mean functional images into a coupled list
"""

mergenode = pe.Node(interface=util.Merge(2, axis='hstack'),
                    name='merge')
preproc.connect(meanfunc2,'out_file', mergenode, 'in1')
preproc.connect(medianval,'out_stat', mergenode, 'in2')


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
preproc.connect(smoothval, 'fwhm', smooth, 'fwhm')
preproc.connect(maskfunc2, 'out_file', smooth, 'in_file')
preproc.connect(medianval, ('out_stat', getbtthresh), smooth, 'brightness_threshold')
preproc.connect(mergenode, ('out', lambda x: [[tuple([val[0],0.75*val[1]])] for val in x]), smooth, 'usans')

"""
Mask the smoothed data with the dilated mask
"""

maskfunc3 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                op_string='-mas'),
                      iterfield=['in_file'],
                      name='maskfunc3')
preproc.connect(smooth, 'smoothed_file', maskfunc3, 'in_file')
preproc.connect(dilatemask, 'out_file', maskfunc3, 'in_file2')


concatnode = pe.Node(interface=util.Merge(2),
                     name='concat')
preproc.connect(maskfunc2,('out_file', lambda x:[x]), concatnode, 'in1')
preproc.connect(maskfunc3,('out_file', lambda x:[x]), concatnode, 'in2')

selectnode = pe.Node(interface=util.Select(),name='select')

preproc.connect(concatnode, 'out', selectnode, 'inlist')

def chooseindex(fwhm):
    if fwhm<1:
        return [0]
    else:
        return [1]
    
preproc.connect(smoothval, ('fwhm', chooseindex), selectnode, 'index')


"""
Scale the median value of the run is set to 10000
"""

meanscale = pe.MapNode(interface=fsl.ImageMaths(suffix='_gms'),
                      iterfield=['in_file','op_string'],
                      name='meanscale')
preproc.connect(selectnode, 'out', meanscale, 'in_file')

"""
Define a function to get the scaling factor for intensity normalization
"""

def getmeanscale(medianvals):
    return ['-mul %.10f'%(10000./val) for val in medianvals]
preproc.connect(medianval, ('out_stat', getmeanscale), meanscale, 'op_string')

"""
Perform temporal highpass filtering on the data
"""

highpass = pe.MapNode(interface=fsl.ImageMaths(suffix='_tempfilt'),
                      iterfield=['in_file'],
                      name='highpass')
preproc.connect(meanscale, 'out_file', highpass, 'in_file')

"""
Generate a mean functional image from the first run
"""

meanfunc3 = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       iterfield=['in_file'],
                      name='meanfunc3')
preproc.connect(highpass, ('out_file', pickfirst), meanfunc3, 'in_file')


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


preproc.connect([(inputnode, surfregister,[('fssubject_id','subject_id'),
                                           ('surf_dir','subjects_dir')]),
                 (meanfunc2, surfregister,[(('out_file',pickfirst),'source_file')]),
                 (motion_correct, art, [('par_file','realignment_parameters')]),
                 (maskfunc2, art, [('out_file','realigned_files')]),
                 (dilatemask, art, [('out_file', 'mask_file')]),
                 ])


"""
Set up model fitting workflow
-----------------------------

"""

modelfit = pe.Workflow(name='modelfit')

"""
Use :class:`nipype.algorithms.modelgen.SpecifyModel` to generate design information.
"""

modelspec = pe.Node(interface=model.SpecifyModel(),  name="modelspec")
modelspec.inputs.concatenate_runs = False

"""
Use :class:`nipype.interfaces.fsl.Level1Design` to generate a run specific fsf
file for analysis
"""

level1design = pe.Node(interface=fsl.Level1Design(), name="level1design")

"""
Use :class:`nipype.interfaces.fsl.FEATModel` to generate a run specific mat
file for use by FILMGLS
"""

modelgen = pe.MapNode(interface=fsl.FEATModel(), name='modelgen',
                      iterfield = ['fsf_file'])

"""
Set the model generation to run everytime. Since the fsf file, which is the
input to modelgen only references the ev files, modelgen will not run if the ev
file contents are changed but the fsf file is untouched.
"""

modelgen.overwrite = True

"""
Use :class:`nipype.interfaces.fsl.FILMGLS` to estimate a model specified by a
mat file and a functional run
"""

modelestimate = pe.MapNode(interface=fsl.FILMGLS(smooth_autocorr=True,
                                                 mask_size=5,
                                                 threshold=1000),
                           name='modelestimate',
                           iterfield = ['design_file','in_file'])

"""
Use :class:`nipype.interfaces.fsl.ContrastMgr` to generate contrast estimates
"""

conestimate = pe.MapNode(interface=fsl.ContrastMgr(), name='conestimate',
                         iterfield = ['tcon_file','stats_dir'])

modelfit.connect([
   (modelspec,level1design,[('session_info','session_info')]),
   (level1design,modelgen,[('fsf_files','fsf_file')]),
   (modelgen,modelestimate,[('design_file','design_file')]),
   (modelgen,conestimate,[('con_file','tcon_file')]),
   (modelestimate,conestimate,[('results_dir','stats_dir')]),
   ])

"""
Set up fixed-effects workflow
-----------------------------

"""

fixed_fx = pe.Workflow(name='fixedfx')

selectnode = pe.Node(interface=util.IdentityInterface(fields=['runs']),
                    name='idselect')

selectnode.iterables = ('runs', [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,1,2],[0,1,2,3]])

copeselect = pe.MapNode(interface=util.Select(), name='copeselect',
                        iterfield=['inlist'])
varcopeselect = pe.MapNode(interface=util.Select(), name='varcopeselect',
                           iterfield=['inlist'])

# Use :class:`nipype.interfaces.fsl.Merge` to merge the copes and
# varcopes for each condition
copemerge    = pe.MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="copemerge")

varcopemerge = pe.MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="varcopemerge")

# Use :class:`nipype.interfaces.fsl.L2Model` to generate subject and
# condition specific level 2 model design files
level2model = pe.Node(interface=fsl.L2Model(),
                      name='l2model')

def num_copes(files):
    return len(files)

"""
Use :class:`nipype.interfaces.fsl.FLAMEO` to estimate a second level
model
"""

flameo = pe.MapNode(interface=fsl.FLAMEO(run_mode='fe'), name="flameo",
                    iterfield=['cope_file','var_cope_file'])

fixed_fx.connect([(selectnode,copeselect,[('runs','index')]),
                  (selectnode,varcopeselect,[('runs','index')]),
                  (selectnode,level2model,[(('runs', num_copes),'num_copes')]),
                  (copeselect,copemerge,[('out','in_files')]),
                  (varcopeselect,varcopemerge,[('out','in_files')]),
                  (varcopeselect,varcopemerge,[('out','in_files')]),
                  (copemerge,flameo,[('merged_file','cope_file')]),
                  (varcopemerge,flameo,[('merged_file','var_cope_file')]),
                  (level2model,flameo, [('design_mat','design_file'),
                                        ('design_con','t_con_file'),
                                        ('design_grp','cov_split_file')]),
                  ])


"""
Set up first-level workflow
---------------------------

"""

def sort_copes(files):
    numelements = len(files[0])
    outfiles = []
    for i in range(numelements):
        outfiles.insert(i,[])
        for j, elements in enumerate(files):
            outfiles[i].append(elements[i])
    return outfiles


"""
Preproc + Analysis + VolumeNormalization workflow
-------------------------------------------------

Connect up the lower level workflows into an integrated analysis. In addition,
we add an input node that specifies all the inputs needed for this
workflow. Thus, one can import this workflow and connect it to their own data
sources. An example with the nifti-tutorial data is provided below.

For this workflow the only necessary inputs are the functional images, a
freesurfer subject id corresponding to recon-all processed data, the session
information for the functional runs and the contrasts to be evaluated.
"""

inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                             'surf_dir',
                                                             'subject_id',
                                                             'fssubject_id',
                                                             'session_info',
                                                             'contrasts']),
                    name='inputnode')

"""
Connect the components into an integrated workflow.
"""

l1pipeline = pe.Workflow(name='firstlevel')
l1pipeline.connect([(inputnode,preproc,[('func','inputspec.func'),
                                        ('fssubject_id','inputspec.fssubject_id'),
                                        ('surf_dir','inputspec.surf_dir')]),
                    (inputnode,modelfit,[('subject_id','modelspec.subject_id'),
                                         ('session_info','modelspec.subject_info'),
                                         ('contrasts','level1design.contrasts'),
                                         ]),
                    (preproc, modelfit, [('highpass.out_file', 'modelspec.functional_runs'),
                                         ('highpass.out_file', 'modelestimate.in_file'),
                                         ('realign.par_file',
                                          'modelspec.realignment_parameters'),
                                         ('art.outlier_files', 'modelspec.outlier_files')]),
                    (preproc, fixed_fx, [('dilatemask.out_file', 'flameo.mask_file')]),
                    (modelfit, fixed_fx,[(('conestimate.copes', sort_copes),'copeselect.inlist'),
                                         (('conestimate.varcopes', sort_copes),'varcopeselect.inlist'),
                                         ])
                    ])


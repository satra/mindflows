import os                                    # system functions

import numpy as np

import nipype.algorithms.modelgen as model   # model generation
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.io as nio           # i/o routines
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine

from nipype.externals.pynifti import load

"""
Setup any package specific configuration. The output file format for FSL
routines is being set to compressed NIFTI.
"""

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from mindflows.gablab.fsl_flow import (preproc, modelfit, overlay,
                                       normalize, applynorm)


"""
Set up fixed-effects workflow
-----------------------------

"""

fixed_fx = pe.Workflow(name='fixedfx')

selectnode = pe.Node(interface=util.IdentityInterface(fields=['runs','funcdata']),
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

ztopval = pe.MapNode(interface=fsl.ImageMaths(op_string='-ztop',
                                              suffix='_pval'),name='ztop',
                     iterfield=['in_file'])
                     
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
                  (flameo,ztopval, [('zstats','in_file')]),
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
                    (preproc,normalize,[("fssource.brainmask", "niftimask.in_file"),
                                        ("fssource.T1", "niftit1.in_file"),
                                        ("surfregister.out_fsl_file", "matconcat.in_file")]),
                    (preproc, modelfit, [('highpass.out_file', 'modelspec.functional_runs'),
                                         ('highpass.out_file', 'modelestimate.in_file'),
                                         ('realign.par_file',
                                          'modelspec.realignment_parameters'),
                                         ('art.outlier_files', 'modelspec.outlier_files')]),
                    # force idselect to get executed after smoothing.
                    (preproc, fixed_fx, [('dilatemask.out_file', 'flameo.mask_file'),
                                         ('highpass.out_file','idselect.funcdata')]),
                    (modelfit, fixed_fx,[(('conestimate.copes', sort_copes),'copeselect.inlist'),
                                         (('conestimate.varcopes', sort_copes),'varcopeselect.inlist'),
                                         ]),
                    (normalize, applynorm, [("fnirt.fieldcoeff_file", "warpfunc.field_file"),
                                            ("matconcat.out_file", "funcxfm.in_matrix_file")]),
                    (preproc, applynorm, [("surfregister.out_fsl_file", "warpfunc.premat")]),
                    (fixed_fx, applynorm, [(("flameo.copes", lambda x:x[0]),
                                             "warpfunc.in_file"),
                                           (("flameo.copes", lambda x:x[0]),
                                             "funcxfm.in_file")]),
                    (preproc, overlay, [('convert2nii.out_file',
                                         'overlaystats.background_image')]),
                    (fixed_fx, overlay, [('flameo.zstats','overlaystats.stat_image')]),
                    ])


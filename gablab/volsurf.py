# Import processing relevant modules
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.io as nio           # i/o routines
import nipype.algorithms.modelgen as model   # model generation
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine


"""
Setup preprocessing workflow
----------------------------

This is a generic preprocessing workflow that can be used by different analyses

"""

preproc = pe.Workflow(name='preproc')


"""Use :class:`nipype.interfaces.spm.Realign` for motion correction
and register all images to the mean image.
"""

realign = pe.Node(interface=spm.Realign(), name="realign")
realign.inputs.register_to_mean = True

"""Use :class:`nipype.algorithms.rapidart` to determine which of the
images in the functional series are outliers based on deviations in
intensity or movement.
"""

art = pe.Node(interface=ra.ArtifactDetect(), name="art")
#art.inputs.use_differences      = [False,True]
#art.inputs.use_norm             = True
#art.inputs.norm_threshold       = 0.5
#art.inputs.zintensity_threshold = 3
art.inputs.mask_type            = 'file'


#run FreeSurfer's BBRegister
surfregister = pe.Node(interface=fs.BBRegister(),name='surfregister')
surfregister.inputs.init = 'fsl'
surfregister.inputs.contrast_type = 't2'

# Get information from the FreeSurfer directories (brainmask, etc)
FreeSurferSource = pe.Node(interface=nio.FreeSurferSource(), name='fssource')

# Allow inversion of brainmask.mgz to volume (functional) space for alignment
ApplyVolTransform = pe.Node(interface=fs.ApplyVolTransform(),
                            name='applyreg')
ApplyVolTransform.inputs.inverse = True 


# Allow for thresholding of volumized brainmask
Threshold = pe.Node(interface=fs.Binarize(),name='threshold')
Threshold.inputs.min = 10

convert2nii = pe.Node(interface=fs.MRIConvert(out_type='nii'),name='convert2nii')

"""Smooth the functional data using
:class:`nipype.interfaces.spm.Smooth`.
"""

volsmooth = pe.Node(interface=spm.Smooth(), name = "volsmooth")
surfsmooth = pe.MapNode(interface=fs.Smooth(proj_frac_avg=(0,1,0.1)), name = "surfsmooth",
                        iterfield=['in_file'])

preproc.connect([(realign, surfregister,[('mean_image', 'source_file')]),
                 (FreeSurferSource, ApplyVolTransform,[('brainmask','target_file')]),
                 (surfregister, ApplyVolTransform,[('out_reg_file','reg_file')]),
                 (realign, ApplyVolTransform,[('mean_image', 'source_file')]),
                 (ApplyVolTransform, Threshold,[('transformed_file','in_file')]),
                 (Threshold, convert2nii, [('binary_file', 'in_file')]),
                 (realign, art,[('realignment_parameters','realignment_parameters'),
                               ('realigned_files','realigned_files')]),
                 (convert2nii, art, [('out_file', 'mask_file')]),
                 (realign, volsmooth, [('realigned_files', 'in_files')]),
                 (realign, surfsmooth, [('realigned_files', 'in_file')]),
                 (surfregister, surfsmooth, [('out_reg_file','reg_file')]),
                 ])


"""
Set up volume analysis workflow
-------------------------------

"""

volanalysis = pe.Workflow(name='volanalysis')

"""Generate SPM-specific design information using
:class:`nipype.interfaces.spm.SpecifyModel`.
"""

modelspec = pe.Node(interface=model.SpecifyModel(), name= "modelspec")
modelspec.inputs.concatenate_runs        = True
modelspec.overwrite = True

"""Generate a first level SPM.mat file for analysis
:class:`nipype.interfaces.spm.Level1Design`.
"""

level1design = pe.Node(interface=spm.Level1Design(), name= "level1design")
level1design.inputs.bases              = {'hrf':{'derivs': [0,0]}}

"""Use :class:`nipype.interfaces.spm.EstimateModel` to determine the
parameters of the model.
"""

level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
level1estimate.inputs.estimation_method = {'Classical' : 1}

"""Use :class:`nipype.interfaces.spm.EstimateContrast` to estimate the
first level contrasts specified in a few steps above.
"""

contrastestimate = pe.Node(interface = spm.EstimateContrast(), name="contrastestimate")

volanalysis.connect([(modelspec,level1design,[('session_info','session_info')]),
                  (level1design,level1estimate,[('spm_mat_file','spm_mat_file')]),
                  (level1estimate,contrastestimate,[('spm_mat_file','spm_mat_file'),
                                                  ('beta_images','beta_images'),
                                                  ('residual_image','residual_image')]),
                  ])

"""
Set up surface analysis workflow
--------------------------------

"""
surfanalysis = volanalysis.clone(name='surfanalysis')


"""
Set up volume normalization workflow
------------------------------------
"""

volnorm = pe.Workflow(name='volnormconimages')

convert = pe.Node(interface=fs.MRIConvert(out_type='nii'),name='convert2nii')
convert2 = pe.MapNode(interface=fs.MRIConvert(in_type='nifti1',out_type='nii'),
                      iterfield=['in_file'],
                      name='convertnifti12nii')
segment = pe.Node(interface=spm.Segment(), name='segment')
normwreg = pe.MapNode(interface=fs.ApplyVolTransform(),
                      iterfield=['source_file'],
                      name='applyreg2con')
normalize = pe.Node(interface=spm.Normalize(jobtype='write'),
                    name='norm2mni')

volnorm.connect([(convert, segment, [('out_file','data')]),
                 (convert2, normwreg, [('out_file','source_file')]),
                 (segment, normalize, [('transformation_mat', 'parameter_file')]),
                 (normwreg, normalize, [('transformed_file','apply_to_files')]),
                 ])

"""
Preproc + Analysis pipeline
---------------------------

"""

inputnode = pe.Node(interface=util.IdentityInterface(fields=['struct',
                                                             'func',
                                                             'subject_id',
                                                             'session_info',
                                                             'contrasts']),
                    name='inputnode')

"""
Use :class:`nipype.algorithms.rapidart` to determine if stimuli are correlated with motion or intensity parameters (STIMULUS CORRELATED MOTION).
"""

stimcorr = pe.Node(interface=ra.StimulusCorrelation(),name='stimcorr')
stimcorr.inputs.concatenated_design             = True

"""
Merge con images and T images into a single list that will then be normalized
"""

mergefiles = pe.Node(interface=util.Merge(2),
                     name='mergeconfiles')


l1pipeline = pe.Workflow(name='firstlevel')
l1pipeline.connect([(inputnode,preproc,[('func','realign.in_files'),
                                        ('subject_id','surfregister.subject_id'),
                                        ('subject_id','fssource.subject_id'),
                                        ]),
                    (inputnode, volanalysis,[('session_info','modelspec.subject_info'),
                                             ('subject_id','modelspec.subject_id'),
                                             ('contrasts','contrastestimate.contrasts')]),
                    (inputnode, surfanalysis,[('session_info','modelspec.subject_info'),
                                              ('subject_id','modelspec.subject_id'),
                                              ('contrasts','contrastestimate.contrasts')]),
                    ])
# attach volume and surface model specification and estimation components
l1pipeline.connect([(preproc, volanalysis, [('realign.realignment_parameters',
                                            'modelspec.realignment_parameters'),
                                           ('volsmooth.smoothed_files',
                                            'modelspec.functional_runs'),
                                           ('art.outlier_files',
                                            'modelspec.outlier_files'),
                                           ('convert2nii.out_file',
                                            'level1design.mask_image')]),
                    (preproc, surfanalysis, [('realign.realignment_parameters',
                                              'modelspec.realignment_parameters'),
                                             ('surfsmooth.smoothed_file',
                                              'modelspec.functional_runs'),
                                             ('art.outlier_files',
                                              'modelspec.outlier_files'),
                                             ('convert2nii.out_file',
                                              'level1design.mask_image')]),
                    (preproc, stimcorr,[('realign.realignment_parameters',
                                         'realignment_parameters'),
                                        ('art.intensity_files','intensity_values')]),
                    (volanalysis, stimcorr, [('level1design.spm_mat_file',
                                              'spm_mat_file')]),
                    ])

# attach volume contrast normalization components
l1pipeline.connect([(preproc, volnorm, [('fssource.orig','convert2nii.in_file'),
                                        ('surfregister.out_reg_file','applyreg2con.reg_file'),
                                        ('fssource.orig','applyreg2con.target_file')]),
                    (volanalysis, mergefiles,[('contrastestimate.con_images','in1'),
                                              ('contrastestimate.spmT_images','in2'),
                                              ]),
                    (mergefiles, volnorm, [('out',
                                            'convertnifti12nii.in_file')]),
                  ])

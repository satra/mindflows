# Import processing relevant modules

import nipype.algorithms.modelgen as model   # model generator
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.utility as util     # utility
import nipype.interfaces.io as nio    # input output nodes
import nipype.interfaces.freesurfer as fs    # freesurfer

import nipype.pipeline.engine as pe          # pypeline engine


def create_spmfspreproc(name='preproc'):
    """
    Setup preprocessing workflow
    ----------------------------

    This is a generic preprocessing workflow that can be used by
    different analyses

    """

    preproc = pe.Workflow(name=name)

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['TR',
                                                                 'TA',
                                                                 'nslices',
                                                                 'slice_order',
                                                                 'ref_slice',
                                                                 'func',
                                                                 'fwhm',
                                                                 'subject_id',
                                                                 'subjects_dir',
                                                                 'output_dir']),
                        name='inputspec')


    """Use :class:`nipype.interfaces.spm.SliceTiming` for correcting
    differences in acquisition of slices
    """

    slicetimecorrect = pe.Node(interface=spm.SliceTiming(),
                               name="slicetimecorrect")

    """Use :class:`nipype.interfaces.spm.Realign` for motion correction
    and register all images to the mean image.
    """

    realign = pe.Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True

    """Smooth the functional data using
    :class:`nipype.interfaces.spm.Smooth`.
    """

    smooth = pe.Node(interface=spm.Smooth(), name = "smooth")

    #run FreeSurfer's BBRegister
#    surfregister = pe.Node(interface=fs.BBRegister(),name='surfregister')
#    surfregister.inputs.init = 'fsl'
#    surfregister.inputs.contrast_type = 't2'

    preproc.connect([(inputnode, slicetimecorrect,[('func','in_files'),
                                                   ('nslices','num_slices'),
                                                   ('TR','time_repetition'),
                                                   ('TA','time_acquisition'),
                                                   ('slice_order','slice_order'),
                                                   ('ref_slice','ref_slice')]),
                     (inputnode, smooth,[('fwhm','fwhm')]),
#                     (inputnode, surfregister, [('subject_id',
#                                                 'subject_id'),
#                                                ('subjects_dir',
#                                                 'subjects_dir')]),
                     (slicetimecorrect, realign,[('timecorrected_files',
                                                  'in_files')]),
                     (realign,smooth,[('realigned_files','in_files')]),
#                     (realign, surfregister,[('mean_image',
#                                              'source_file')]),
                     ])
    
    sink1 = pe.Node(nio.XNATSink(), name='sink')
    sink1.inputs.xnat_server = 'https://imagen.cea.fr/imagen_database'
    sink1.inputs.xnat_user = 'ys218403'
    sink1.inputs.xnat_pwd = 'y4q8c7b7p4'
    sink1.inputs.project_id = 'metabase_ys218403'

    preproc.connect(inputnode, 'subject_id', sink1, 'container')
    preproc.connect(inputnode, 'output_dir', sink1, 'base_directory')
    preproc.connect(realign, 'realignment_parameters', sink1, 'preproc.realign')
    preproc.connect(smooth, 'smoothed_files', sink1, 'preproc.func')
#    preproc.connect(surfregister, 'out_reg_file', sink1, 'preproc.reg')
#    preproc.connect(surfregister, 'min_cost_file', sink1, 'preproc.reg.@mincost')

    return preproc

def create_analysis(name='firstlevel'):
    volanalysis = pe.Workflow(name=name)
    """Generate SPM-specific design information using
    :class:`nipype.interfaces.spm.SpecifyModel`.
    """

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['subject_info',
                                                                 'func',
                                                                 'TR',
                                                                 'hpcutoff',
                                                                 'contrasts',
                                                                 'subject_id',
                                                                 'realignment_parameters',
                                                                 'output_dir']),
                        name='inputspec')
    modelspec = pe.Node(interface=model.SpecifyModel(), name= "modelspec")
    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.output_units = 'secs'
    
    """Generate a first level SPM.mat file for analysis
    :class:`nipype.interfaces.spm.Level1Design`.
    """

    level1design = pe.Node(interface=spm.Level1Design(), name= "level1design")
    level1design.inputs.bases              = {'hrf':{'derivs': [0,0]}}
    level1design.inputs.timing_units       = modelspec.inputs.output_units

    """Use :class:`nipype.interfaces.spm.EstimateModel` to determine the
    parameters of the model.
    """

    level1estimate = pe.Node(interface=spm.EstimateModel(),
                             name="level1estimate")
    level1estimate.inputs.estimation_method = {'Classical' : 1}

    """Use :class:`nipype.interfaces.spm.EstimateContrast` to estimate the
    first level contrasts specified in a few steps above.
    """

    contrastestimate = pe.Node(interface = spm.EstimateContrast(),
                               name="contrastestimate")

    volanalysis.connect([(inputnode,modelspec,[('subject_info','subject_info'),
                                               ('func','functional_runs'),
                                               ('TR','time_repetition'),
                                               ('realignment_parameters','realignment_parameters'),
                                               ('hpcutoff','high_pass_filter_cutoff')]),
                         (modelspec,level1design,[('session_info',
                                                   'session_info')]),
                         (inputnode, level1design, [('TR','interscan_interval'),
                                                  ]),
                         (level1design,level1estimate,[('spm_mat_file',
                                                        'spm_mat_file')]),
                         (inputnode,contrastestimate,[('contrasts','contrasts')]),
                         (level1estimate,contrastestimate,[('spm_mat_file',
                                                            'spm_mat_file'),
                                                           ('beta_images',
                                                            'beta_images'),
                                                           ('residual_image',
                                                            'residual_image'
                                                            )]),
                         ])
    sink1 = pe.Node(nio.DataSink(parameterization=False), name='sink')
#    volanalysis.connect(inputnode, 'subject_id', sink1, 'container')
#    volanalysis.connect(inputnode, 'output_dir', sink1, 'base_directory')



    volanalysis.connect(inputnode, 'subject_id', sink1, 'subject_id')
    volanalysis.connect(inputnode, 'output_dir', sink1, 'experiment_id')
    volanalysis.connect(contrastestimate, 'con_images', sink1, 'estimate.@con')
    volanalysis.connect(contrastestimate, 'spmT_images', sink1, 'estimate.@spmt')
    return volanalysis

def create_surfanalysis(name='surfanalysis'):
    l2flow = pe.Workflow(name=name)
    

    l2inputnode = pe.Node(interface=util.IdentityInterface(fields=['contuples','hemi']),
                          name='inputspec')
    l2inputnode.iterables = ('hemi', ['lh','rh'])

    """
    Concatenate contrast images projected to fsaverage
    """

    l2concat = pe.Node(interface=fs.MRISPreproc(), name='concat')
    l2concat.inputs.proj_frac = 0.5
    l2concat.inputs.target = 'fsaverage'
    l2concat.inputs.fwhm = 5
    l2flow.connect(l2inputnode, 'contuples', l2concat, 'vol_measure_file')
    l2flow.connect(l2inputnode, 'hemi', l2concat, 'hemi')

    """
    Perform a one sample t-test
    """

    l2ttest = pe.Node(interface=fs.OneSampleTTest(), name='onesample')
    l2flow.connect(l2inputnode, ('hemi', lambda x: (l2concat.inputs.target, x, 'white')), l2ttest, 'surf')
    l2flow.connect(l2concat, 'out_file', l2ttest, 'in_file')

    return l2flow

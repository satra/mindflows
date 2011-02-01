import os                                    # system functions

from warnings import warn

import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.spm as spm          # freesurfer
import nipype.interfaces.io as nio           # i/o routines
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine

from nibabel import load


warn('WORK IN PROGRESS. USE WITH CAUTION')


"""
Setup any package specific configuration. The output file format for FSL
routines is being set to compressed NIFTI.
"""

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')


"""
Set up FSL preprocessing workflow
--------------------

In this tutorial we will be setting up a hierarchical workflow for fsl
analysis. This will demonstrate how pre-defined workflows can be setup and
shared across users, projects and labs.


Setup preprocessing workflow
----------------------------

This is a generic fsl feat preprocessing workflow encompassing skull stripping,
motion correction and smoothing operations.

"""

def create_featpreproc(name='featpreproc'):
    """Create a FEAT preprocessing workflow
    
    Parameters
    ----------
    
    func : functional runs (filename or list of filenames)
    fwhm : fwhm for smoothing with SUSAN
    highpass : HWHM in TRs
    outdir : where preprocessed data should be stored
    subjectid : subjectid (used for storing output under subject's name
    subs : paths substitutions for datasink

    Example
    -------

    >>> from mindflows.gablab.preproc_schemes import create_featpreproc
    >>> import os
    >>> preproc = create_featpreproc()
    >>> preproc.inputs.inputspec.func = 'f3.nii'
    >>> preproc.inputs.inputspec.fwhm = 5
    >>> preproc.inputs.inputspec.highpass = 128./(2*2.5)
    >>> preproc.inputs.inputspec.outdir = os.path.abspath('l1out')
    >>> preproc.inputs.inputspec.subjectid = 's1'
    >>> preproc.inputs.inputspec.subs = []
    >>> preproc.base_dir = '/tmp'
    >>> preproc.run() # doctest: +SKIP
    
    """
    
    featpreproc = pe.Workflow(name=name)

    """
    Set up a node to define all inputs required for the preprocessing workflow

    """

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'fwhm',
                                                                 'highpass',
                                                                 'outdir',
                                                                 'subjectid',
                                                                 'subs']),
                        name='inputspec')

    """
    Create a datasink 
    """
    
    datasink = pe.Node(interface=nio.DataSink(),
                       name='datasink')
    featpreproc.connect(inputnode, 'outdir', datasink, 'base_directory')
    featpreproc.connect(inputnode, 'subjectid', datasink, 'container')
    featpreproc.connect(inputnode, 'subs', datasink, 'substitutions')
    
    """
    Convert functional images to float representation. Since there can
    be more than one functional run we use a MapNode to convert each
    run.
    """

    img2float = pe.MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                                 op_string = '',
                                                 suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')
    featpreproc.connect(inputnode, 'func', img2float, 'in_file')

    """
    Extract the first volume of the first run as the reference
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

    featpreproc.connect(img2float, ('out_file', pickfirst), extract_ref, 'in_file')
    featpreproc.connect(extract_ref, 'roi_file', datasink, 'reference')

    """
    Realign the functional runs to the reference (1st volume of first run)
    """

    motion_correct = pe.MapNode(interface=fsl.MCFLIRT(save_mats = True,
                                                      save_plots = True),
                                name='realign',
                                iterfield = ['in_file'])
    featpreproc.connect(img2float, 'out_file', motion_correct, 'in_file')
    featpreproc.connect(extract_ref, 'roi_file', motion_correct, 'ref_file')
    featpreproc.connect(motion_correct, 'par_file', datasink, 'motion.parameters')
    featpreproc.connect(motion_correct, 'out_file', datasink, 'motion.realigned')

    """
    Plot the estimated motion parameters
    """

    plot_motion = pe.MapNode(interface=fsl.PlotMotionParams(in_source='fsl'),
                            name='plot_motion',
                            iterfield=['in_file'])
    plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
    featpreproc.connect(motion_correct, 'par_file', plot_motion, 'in_file')
    featpreproc.connect(plot_motion, 'out_file', datasink, 'motion.plots')

    """
    Extract the mean volume of the first functional run
    """

    meanfunc = pe.Node(interface=fsl.ImageMaths(op_string = '-Tmean',
                                                suffix='_mean'),
                       name='meanfunc')
    featpreproc.connect(motion_correct, ('out_file', pickfirst), meanfunc, 'in_file')

    """
    Strip the skull from the mean functional to generate a mask
    """

    meanfuncmask = pe.Node(interface=fsl.BET(mask = True,
                                             no_output=True,
                                             frac = 0.3),
                           name = 'meanfuncmask')
    featpreproc.connect(meanfunc, 'out_file', meanfuncmask, 'in_file')

    """
    Mask the functional runs with the extracted mask
    """

    maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_bet',
                                                   op_string='-mas'),
                          iterfield=['in_file'],
                          name = 'maskfunc')
    featpreproc.connect(motion_correct, 'out_file', maskfunc, 'in_file')
    featpreproc.connect(meanfuncmask, 'mask_file', maskfunc, 'in_file2')


    """
    Determine the 2nd and 98th percentile intensities of each functional run
    """

    getthresh = pe.MapNode(interface=fsl.ImageStats(op_string='-p 2 -p 98'),
                           iterfield = ['in_file'],
                           name='getthreshold')
    featpreproc.connect(maskfunc, 'out_file', getthresh, 'in_file')


    """
    Threshold the first run of the functional data at 10% of the 98th percentile
    """

    threshold = pe.Node(interface=fsl.ImageMaths(out_data_type='char',
                                                 suffix='_thresh'),
                           name='threshold')
    featpreproc.connect(maskfunc, ('out_file', pickfirst), threshold, 'in_file')

    """
    Define a function to get 10% of the intensity
    """

    def getthreshop(thresh):
        return '-thr %.10f -Tmin -bin'%(0.1*thresh[0][1])
    featpreproc.connect(getthresh, ('out_stat', getthreshop), threshold, 'op_string')

    """
    Determine the median value of the functional runs using the mask
    """

    medianval = pe.MapNode(interface=fsl.ImageStats(op_string='-k %s -p 50'),
                           iterfield = ['in_file'],
                           name='medianval')
    featpreproc.connect(motion_correct, 'out_file', medianval, 'in_file')
    featpreproc.connect(threshold, 'out_file', medianval, 'mask_file')

    """
    Dilate the mask
    """

    dilatemask = pe.Node(interface=fsl.ImageMaths(suffix='_dil',
                                                  op_string='-dilF'),
                           name='dilatemask')
    featpreproc.connect(threshold, 'out_file', dilatemask, 'in_file')
    featpreproc.connect(dilatemask, 'out_file', datasink, 'mask')

    """
    Mask the motion corrected functional runs with the dilated mask
    """

    maskfunc2 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                    op_string='-mas'),
                          iterfield=['in_file'],
                          name='maskfunc2')
    featpreproc.connect(motion_correct, 'out_file', maskfunc2, 'in_file')
    featpreproc.connect(dilatemask, 'out_file', maskfunc2, 'in_file2')

    """
    Determine the mean image from each functional run
    """

    meanfunc2 = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                    suffix='_mean'),
                           iterfield=['in_file'],
                           name='meanfunc2')
    featpreproc.connect(maskfunc2, 'out_file', meanfunc2, 'in_file')

    """
    Merge the median values with the mean functional images into a coupled list
    """

    mergenode = pe.Node(interface=util.Merge(2, axis='hstack'),
                        name='merge')
    featpreproc.connect(meanfunc2,'out_file', mergenode, 'in1')
    featpreproc.connect(medianval,'out_stat', mergenode, 'in2')


    """
    Smooth each run using SUSAN with the brightness threshold set to 75%
    of the median value for each run and a mask consituting the mean
    functional
    """

    smooth = pe.MapNode(interface=fsl.SUSAN(),
                        iterfield=['in_file', 'brightness_threshold','usans'],
                        name='smooth')

    """
    Define a function to get the brightness threshold for SUSAN
    """

    def getbtthresh(medianvals):
        return [0.75*val for val in medianvals]
    featpreproc.connect(inputnode, 'fwhm', smooth, 'fwhm')
    featpreproc.connect(maskfunc2, 'out_file', smooth, 'in_file')
    featpreproc.connect(medianval, ('out_stat', getbtthresh), smooth, 'brightness_threshold')
    featpreproc.connect(mergenode, ('out', lambda x: [[tuple([val[0],0.75*val[1]])] for val in x]), smooth, 'usans')

    """
    Mask the smoothed data with the dilated mask
    """

    maskfunc3 = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                    op_string='-mas'),
                          iterfield=['in_file'],
                          name='maskfunc3')
    featpreproc.connect(smooth, 'smoothed_file', maskfunc3, 'in_file')
    featpreproc.connect(dilatemask, 'out_file', maskfunc3, 'in_file2')


    concatnode = pe.Node(interface=util.Merge(2),
                         name='concat')
    featpreproc.connect(maskfunc2,('out_file', lambda x:[x]), concatnode, 'in1')
    featpreproc.connect(maskfunc3,('out_file', lambda x:[x]), concatnode, 'in2')

    """
    The following nodes select smooth or unsmoothed data depending on the
    fwhm. This is because SUSAN defaults to smoothing the data with about the
    voxel size of the input data if the fwhm parameter is less than 1/3 of the
    voxel size.
    """
    selectnode = pe.Node(interface=util.Select(),name='select')

    featpreproc.connect(concatnode, 'out', selectnode, 'inlist')

    def chooseindex(fwhm):
        if fwhm<1:
            return [0]
        else:
            return [1]

    featpreproc.connect(inputnode, ('fwhm', chooseindex), selectnode, 'index')
    featpreproc.connect(selectnode, 'out', datasink, 'smoothed')


    """
    Scale the median value of the run is set to 10000
    """

    meanscale = pe.MapNode(interface=fsl.ImageMaths(suffix='_gms'),
                          iterfield=['in_file','op_string'],
                          name='meanscale')
    featpreproc.connect(selectnode, 'out', meanscale, 'in_file')

    """
    Define a function to get the scaling factor for intensity normalization
    """

    def getmeanscale(medianvals):
        return ['-mul %.10f'%(10000./val) for val in medianvals]
    featpreproc.connect(medianval, ('out_stat', getmeanscale), meanscale, 'op_string')

    """
    Perform temporal highpass filtering on the data
    """

    highpass = pe.MapNode(interface=fsl.ImageMaths(suffix='_tempfilt'),
                          iterfield=['in_file'],
                          name='highpass')
    featpreproc.connect(inputnode, ('highpass', lambda x:'-bptf %.10f -1'%x), highpass, 'op_string')
    featpreproc.connect(meanscale, 'out_file', highpass, 'in_file')
    featpreproc.connect(highpass, 'out_file', datasink, 'highpassed')

    """
    Generate a mean functional image from the first run
    """

    meanfunc3 = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                    suffix='_mean'),
                           iterfield=['in_file'],
                          name='meanfunc3')
    featpreproc.connect(highpass, ('out_file', pickfirst), meanfunc3, 'in_file')
    featpreproc.connect(meanfunc3, 'out_file', datasink, 'mean')

    
    return featpreproc

def create_spmpreproc1(name='spmpreproc'):
    """Use SPM to do realignment and smoothing
    
    """
    preproc = pe.Workflow(name=name)

    """
    Set up a node to define all inputs required for the preprocessing workflow

    """

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'fwhm',
                                                                 'outdir',
                                                                 'subjectid',
                                                                 'subs']),
                        name='inputspec')

    """
    Create a datasink 
    """
    
    datasink = pe.Node(interface=nio.DataSink(),
                       name='datasink')
    preproc.connect(inputnode, 'outdir', datasink, 'base_directory')
    preproc.connect(inputnode, 'subjectid', datasink, 'container')
    preproc.connect(inputnode, 'subs', datasink, 'substitutions')

    """
    Use :class:`nipype.interfaces.spm.Realign` for motion correction and
    register all images to the mean image.
    """

    realign = pe.Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True
    preproc.connect(inputnode,'func', realign, 'in_files')
    preproc.connect(realign, 'realigned_files', datasink, 'motion.realigned')
    preproc.connect(realign, 'realignment_parameters', datasink, 'motion.parameters')

    """
    Plot the estimated motion parameters
    """

    plot_motion = pe.MapNode(interface=fsl.PlotMotionParams(in_source='spm'),
                            name='plot_motion',
                            iterfield=['in_file'])
    plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
    preproc.connect(realign, 'realignment_parameters', plot_motion, 'in_file')
    preproc.connect(plot_motion, 'out_file', datasink, 'motion.plots')
    
    """
    Smooth the functional data using :class:`nipype.interfaces.spm.Smooth`.
    """

    smooth = pe.Node(interface=spm.Smooth(), name = "smooth")
    preproc.connect(inputnode, 'fwhm', smooth, 'fwhm')
    preproc.connect(realign, 'realigned_files', smooth, 'in_files')
    preproc.connect(smooth, 'smoothed_files', datasink, 'smoothed')
    
    return preproc


def create_spmpreproc2(name='spmpreproc'):
    preproc = pe.Workflow(name=name)

    """
    Set up a node to define all inputs required for the preprocessing workflow

    """

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'struct',
                                                                 'fwhm',
                                                                 'outdir',
                                                                 'subjectid',
                                                                 'subs']),
                        name='inputspec')

    """
    Create a datasink 
    """
    
    datasink = pe.Node(interface=nio.DataSink(),
                       name='datasink')
    preproc.connect(inputnode, 'outdir', datasink, 'base_directory')
    preproc.connect(inputnode, 'subjectid', datasink, 'container')
    preproc.connect(inputnode, 'subs', datasink, 'substitutions')

    """
    Use :class:`nipype.interfaces.spm.Realign` for motion correction and
    register all images to the mean image.
    """

    realign = pe.Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True
    preproc.connect(inputnode,'func', realign, 'in_files')
    preproc.connect(realign, 'realignment_parameters', datasink, 'motion.parameters')

    """
    Plot the estimated motion parameters
    """

    plot_motion = pe.MapNode(interface=fsl.PlotMotionParams(in_source='spm'),
                            name='plot_motion',
                            iterfield=['in_file'])
    plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
    preproc.connect(realign, 'realignment_parameters', plot_motion, 'in_file')
    preproc.connect(plot_motion, 'out_file', datasink, 'motion.plots')
    
    """
    Use :class:`nipype.interfaces.spm.Coregister` to perform a rigid body
    registration of the functional data to the structural data.
    """

    coregister = pe.Node(interface=spm.Coregister(), name="coregister")
    coregister.inputs.jobtype = 'estimate'
    preproc.connect(inputnode, 'struct', coregister, 'target')
    preproc.connect(realign, 'mean_image', coregister,'source')
    preproc.connect(realign, 'realigned_files', coregister, 'apply_to_files')

    segment = pe.Node(interface=spm.Segment(), name='segment')
    preproc.connect(inputnode, 'struct', segment, 'data')
    preproc.connect(segment, 'transformation_mat', datasink, 'segment.@transform')

    """
    Warp functional and structural data to SPM's T1 template using
    :class:`nipype.interfaces.spm.Normalize`.  The tutorial data set includes
    the template image, T1.nii.
    """

    normalize = pe.Node(interface=spm.Normalize(), name = "normalize")
    normalize.inputs.jobtype = 'write'
    preproc.connect(coregister, 'coregistered_files', normalize, 'apply_to_files') 
    preproc.connect(segment, 'transformation_mat', normalize, 'parameter_file')
    preproc.connect(normalize, 'normalized_files', datasink, 'normalized')

    """
    Smooth the functional data using :class:`nipype.interfaces.spm.Smooth`.
    """

    smooth = pe.Node(interface=spm.Smooth(), name = "smooth")
    preproc.connect(inputnode, 'fwhm', smooth, 'fwhm')
    preproc.connect(normalize, 'normalized_files', smooth, 'in_files')
    preproc.connect(smooth, 'smoothed_files', datasink, 'smoothed')
    
    return preproc

def create_spmpreproc3(name='spmpreproc'):
    """ use freesurfer for smoothing and registration
    realignment, coregistration with surface and surface-based smoothing.
    """
    
    preproc = pe.Workflow(name=name)

    """
    Set up a node to define all inputs required for the preprocessing workflow

    """

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'fsid',
                                                                 'fsdir'
                                                                 'fwhm',
                                                                 'outdir',
                                                                 'subjectid',
                                                                 'subs']),
                        name='inputspec')

    """
    Create a datasink 
    """
    
    datasink = pe.Node(interface=nio.DataSink(),
                       name='datasink')
    preproc.connect(inputnode, 'outdir', datasink, 'base_directory')
    preproc.connect(inputnode, 'subjectid', datasink, 'container')
    preproc.connect(inputnode, 'subs', datasink, 'substitutions')

    """
    Use :class:`nipype.interfaces.spm.Realign` for motion correction and
    register all images to the mean image.
    """

    realign = pe.Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True
    realign.inputs.jobtype = 'estimate'
    preproc.connect(inputnode,'func', realign, 'in_files')
    preproc.connect(realign, 'realignment_parameters', datasink, 'motion.parameters')

    """
    Smooth the functional data using :class:`nipype.interfaces.spm.Smooth`.
    """

    smooth = pe.Node(interface=spm.Smooth(), name = "smooth")
    preproc.connect(inputnode, 'fwhm', smooth, 'fwhm')
    preproc.connect(normalize, 'normalized_files', smooth, 'in_files')
    preproc.connect(smooth, 'smoothed_files', datasink, 'smoothed')
    
    return preproc

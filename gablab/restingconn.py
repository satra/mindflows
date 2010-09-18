# Import processing relevant modules
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine

#import nipype.interfaces.fsl as fsl          # fsl



"""
Setup preprocessing workflow
----------------------------

This is a generic preprocessing workflow that can be used by different analyses

"""

restpreproc = pe.Workflow(name='restpreproc')


"""Use :class:`nipype.interfaces.spm.SliceTiming` for correcting
differences in acquisition of slices
"""

slicetimecorrect = pe.Node(interface=spm.SliceTiming(), name="slicetimecorrect")

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
art.inputs.use_differences      = [False,True]
art.inputs.use_norm             = True
art.inputs.norm_threshold       = 0.5
art.inputs.zintensity_threshold = 3
art.inputs.mask_type            = 'spm_global'
art.inputs.parameter_source     = 'SPM'


"""Use :class:`nipype.interfaces.spm.Coregister` to perform a rigid
body registration of the functional data to the structural data.
"""

coregister = pe.Node(interface=spm.Coregister(), name="coregister")
coregister.inputs.jobtype = 'estimate'


"""Use :class:`nipype.interfaces.spm.Segment` to perform segmentation of
the structural image
"""
segment = pe.Node(interface=spm.Segment(), name='segment')
segment.inputs.gm_output_type = [True, True, True]
segment.inputs.wm_output_type = [True, True, False]
segment.inputs.csf_output_type = [True, True, False]

"""Warp functional and structural data to SPM's T1 template using
:class:`nipype.interfaces.spm.Normalize`.  The tutorial data set
includes the template image, T1.nii.
"""

normalize = pe.Node(interface=spm.Normalize(), name = "normalize")
normalize.inputs.jobtype = 'write'

structnormalize = pe.Node(interface=spm.Normalize(), name = "structnormalize")
structnormalize.inputs.jobtype = 'write'

roinormalize = pe.Node(interface=spm.Normalize(), name = "roinormalize")
roinormalize.inputs.jobtype = 'write'
roinormalize.inputs.write_interp = 0 # do nearest neighbor interpolation

"""Smooth the functional data using
:class:`nipype.interfaces.spm.Smooth`.
"""

smooth = pe.Node(interface=spm.Smooth(), name = "smooth")
                 
restpreproc.connect([(slicetimecorrect, realign,[('timecorrected_files','in_files')]),
                 (realign,coregister,[('mean_image', 'source'),
                                      ('realigned_files','apply_to_files')]),
                 (segment, normalize,[('transformation_mat', 'parameter_file')]), 
                 (segment, roinormalize,[('transformation_mat', 'parameter_file')]), 
                 (segment, structnormalize,[('transformation_mat', 'parameter_file')]), 
                 (coregister, normalize, [('coregistered_files','apply_to_files')]),
                 (normalize, smooth, [('normalized_files', 'in_files')]),
                 (realign,art,[('realignment_parameters','realignment_parameters')]),
                 (normalize,art,[('normalized_files','realigned_files')]),
                 ])

coregister2 = pe.Node(interface=spm.Coregister(), name="coregister")

restpreproc2 = pe.Workflow(name='restpreproc')
restpreproc2.connect([(slicetimecorrect, realign,[('timecorrected_files','in_files')]),
                      (realign,coregister2,[('mean_image', 'source'),
                                            ('realigned_files','apply_to_files')]),
                      (realign,art,[('realigned_files','realigned_files'),
                                    ('realignment_parameters','realignment_parameters')]),
                 ])

import os                                    # operating system functions

import nipype.interfaces.io as nio           # Data i/o 
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.fsl as fsl          # fsl
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.utility as util     # misc. modules

"""
Level2 surface-based pipeline
-----------------------------

Create a level2 workflow
"""

def orderfiles(files, subj_list):
    outlist = []
    for s in subj_list:
        for f in files:
            if '%s/'%s in f:
                outlist.append(f)
                continue
    return outlist

def L2FLAME(name='flame'):
    flameflow = pe.Workflow(name=name)
    '''
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['copes',
                                                                 'varcopes',
                                                                 'design',
                                                                 'tconfile',
                                                                 'groupfile',
                                                                 'inferoutliers',
                                                                 'stat_thresh']),
                        name='inputnode')
    '''
    mni152 = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    flamemask = pe.Node(fs.Binarize(min=10, in_file=mni152), name="flamemask")
    flame = pe.Node(fsl.FLAMEO(run_mode="flame1"), name="flame")
    mni_brain = fsl.Info.standard_image("avg152T1.nii.gz")
    overlayflame = pe.MapNode(interface=fsl.Overlay(stat_thresh=(2.5, 5),
                                                 auto_thresh_bg=True,
                                                 show_negative_stats=True,
                                                 background_image=mni_brain),
                              iterfield=['stat_image'],
                              name='overlayflame')

    sliceflame = pe.MapNode(interface=fsl.Slicer(all_axial=True,
                                              image_width=750),
                            iterfield=['in_file'],
                            name='sliceflame')
    
    flameflow.connect([(flamemask,flame, [("binary_file", "mask_file")]),
                       (flame,overlayflame,[('tstats','stat_image')]),
                       (overlayflame,sliceflame,[('out_file', 'in_file')]),
                       ])

    return flameflow

def create_l2fsflow():
    l2fsflow = pe.Workflow(name='l2vol')

    """
    Setup a dummy node to iterate over contrasts and hemispheres
    """

    l2inputnode = pe.Node(interface=util.IdentityInterface(fields=['contrasts',
                                                                   'subjects',
                                                                   'base_directory',
                                                                   'field_template']),
                          name='inputnode')

    """
    Use a datagrabber node to collect contrast images and registration files
    """

    l2source = pe.Node(interface=nio.DataGrabber(infields=['con_id'],
                                                 outfields=['con']),
                       name='l2source')
    l2source.inputs.template = '*'
    l2source.inputs.template_args = dict(con=[['con_id']],reg=[[]])

    l2fsflow.connect(l2inputnode, 'contrasts', l2source, 'con_id')
    l2fsflow.connect(l2inputnode, 'base_directory', l2source, 'base_directory')
    l2fsflow.connect(l2inputnode, 'field_template', l2source, 'field_template')


    """
    Merge contrast images and registration files
    """

    mergesubjnode = pe.Node(interface=util.Merge(2),
                        name='mergesubjects')

    def ordersubjects(data):
        files = []
        subj_list = []
        con_list = []
        for f in data:
            if os.path.exists(f):
                con_list.append(f)
            else:
                subj_list.append(f)
        ordered_subj = orderfiles(con_list, subj_list)
        return ordered_subj

    l2fsflow.connect(l2source,'con', mergesubjnode, 'in1')
    l2fsflow.connect(l2inputnode, 'subjects', mergesubjnode, 'in2')

    """
    Concatenate contrast images projected to fsaverage
    """

    copemerge = pe.Node(fsl.Merge(dimension="t"), name="copemerge")
    l2fsflow.connect(mergesubjnode, ('out', ordersubjects), copemerge, 'in_files')

    """
    Perform a one sample t-test
    """

    l2ttest = pe.Node(interface=fs.OneSampleTTest(), name='onesample')
    l2fsflow.connect(copemerge, 'merged_file', l2ttest, 'in_file')

    mni152 = fsl.Info.standard_image("avg152T1_brain.nii.gz")

    flamemask = pe.Node(fs.Binarize(min=10, in_file=mni152), name="flamemask")

    l2fsflow.connect(flamemask, 'binary_file', l2ttest, "mask_file")

    return l2fsflow

l2fsflow = create_l2fsflow()

def create_l2fslflow():
    l2fslflow = pe.Workflow(name='l2vol')

    """
    Setup a dummy node to iterate over contrasts and hemispheres
    """

    l2inputnode = pe.Node(interface=util.IdentityInterface(fields=['contrasts',
                                                                   'subjects',
                                                                   'base_directory',
                                                                   'field_template']),
                          name='inputnode')

    """
    Use a datagrabber node to collect contrast images and registration files
    """

    l2source = pe.Node(interface=nio.DataGrabber(infields=['con_id'],
                                                 outfields=['cope','varcope']),
                       name='l2source')
    l2source.inputs.template = '*'
    l2source.inputs.template_args = dict(cope=[['con_id']],varcope=[['con_id']])

    l2fslflow.connect(l2inputnode, 'contrasts', l2source, 'con_id')
    l2fslflow.connect(l2inputnode, 'base_directory', l2source, 'base_directory')
    l2fslflow.connect(l2inputnode, 'field_template', l2source, 'field_template')


    def ordersubjects(data):
        files = []
        subj_list = []
        file_list = []
        for f in data:
            if os.path.exists(f):
                file_list.append(f)
            else:
                subj_list.append(f)
        ordered_files = orderfiles(file_list, subj_list)
        return ordered_files

    """
    Merge contrast images and registration files
    """

    mergecopesubj = pe.Node(interface=util.Merge(2),
                            name='mergecopesubj')
    mergevarcopesubj =  pe.Node(interface=util.Merge(2),
                            name='mergevarcopesubj')
    
    l2fslflow.connect(l2source,'cope', mergecopesubj, 'in1')
    l2fslflow.connect(l2inputnode, 'subjects', mergecopesubj, 'in2')

    l2fslflow.connect(l2source,'varcope', mergevarcopesubj, 'in1')
    l2fslflow.connect(l2inputnode, 'subjects', mergevarcopesubj, 'in2')
    
    """
    Concatenate contrast images projected to fsaverage
    """

    copemerge = pe.Node(fsl.Merge(dimension="t"), name="copemerge")
    l2fslflow.connect(mergecopesubj, ('out', ordersubjects), copemerge, 'in_files')
    varcopemerge = pe.Node(fsl.Merge(dimension="t"), name="varcopemerge")
    l2fslflow.connect(mergevarcopesubj, ('out', ordersubjects), varcopemerge, 'in_files')

    
    """
    Perform a one sample t-test
    """
    design = pe.Node(fsl.L2Model(), name="design")
    l2flame = L2FLAME(name='onesample')
    l2fslflow.connect([(l2inputnode, design, [(("subjects", lambda x: len(x)), "num_copes")]),
                       (copemerge,l2flame,[('merged_file','flame.cope_file')]),
                       (varcopemerge,l2flame,[('merged_file','flame.var_cope_file')]),
                       (design,l2flame, [('design_mat','flame.design_file'),
                                       ('design_con','flame.t_con_file'),
                                       ('design_grp','flame.cov_split_file')]),
                       ])

    return l2fslflow


l2fslflow = create_l2fslflow()

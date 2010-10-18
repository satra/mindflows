import os                                    # operating system functions

import nipype.interfaces.io as nio           # Data i/o 
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.utility as util     # misc. modules

"""
Level2 surface-based pipeline
-----------------------------

Create a level2 workflow
"""

l2flow = pe.Workflow(name='l2surf')

"""
Setup a dummy node to iterate over contrasts and hemispheres
"""

l2inputnode = pe.Node(interface=util.IdentityInterface(fields=['contrasts',
                                                               'hemi',
                                                               'subjects',
                                                               'base_directory',
                                                               'field_template']),
                      name='inputnode')

"""
Use a datagrabber node to collect contrast images and registration files
"""

l2source = pe.Node(interface=nio.DataGrabber(infields=['con_id'],
                                             outfields=['con','reg']),
                   name='l2source')
l2source.inputs.template = '*'
l2source.inputs.template_args = dict(con=[['con_id']],reg=[[]])

l2flow.connect(l2inputnode, 'contrasts', l2source, 'con_id')
l2flow.connect(l2inputnode, 'base_directory', l2source, 'base_directory')
l2flow.connect(l2inputnode, 'field_template', l2source, 'field_template')


"""
Merge contrast images and registration files
"""

mergesubjnode = pe.Node(interface=util.Merge(3),
                    name='mergesubjects')

def orderfiles(files, subj_list):
    outlist = []
    for s in subj_list:
        for f in files:
            if '%s/'%s in f:
                outlist.append(f)
                continue
    return outlist

def ordersubjects(data):
    files = []
    subj_list = []
    con_list = []
    reg_list = []
    for f in data:
        if os.path.exists(f):
            if f.endswith('dat'):
                reg_list.append(f)
            else:
                con_list.append(f)
        else:
            subj_list.append(f)
    ordered_subj = orderfiles(con_list, subj_list)
    ordered_reg = orderfiles(reg_list, subj_list)
    filelist= [(v,ordered_reg[i]) for i,v in enumerate(ordered_subj)]
    return filelist

l2flow.connect(l2source,'con', mergesubjnode, 'in1')
l2flow.connect(l2source,'reg', mergesubjnode, 'in2')
l2flow.connect(l2inputnode, 'subjects', mergesubjnode, 'in3')

"""
Concatenate contrast images projected to fsaverage
"""

l2concat = pe.Node(interface=fs.MRISPreproc(), name='concat')
l2concat.inputs.proj_frac = 0.5
l2concat.inputs.target = 'fsaverage'
l2concat.inputs.fwhm = 5

l2flow.connect(l2inputnode, 'hemi', l2concat, 'hemi')
l2flow.connect(mergesubjnode, ('out', ordersubjects), l2concat, 'vol_measure_file')

"""
Perform a one sample t-test
"""

l2ttest = pe.Node(interface=fs.OneSampleTTest(), name='onesample')
l2flow.connect(l2inputnode, ('hemi', lambda x: (l2concat.inputs.target, x, 'white')), l2ttest, 'surf')
l2flow.connect(l2concat, 'out_file', l2ttest, 'in_file')

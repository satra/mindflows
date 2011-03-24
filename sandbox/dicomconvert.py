#! /usr/bin/env python

helpdoc = """Convert dicom TimTrio dirs to nii.gz files based on config files.

This function uses FreeSurfer tools (unpacksdcmdir) to convert Siemens
TrioTim directories. It then proceeeds by extracting dicominfo from each
subject and writing a config file $subject_id/$subject_id.auto.txt in
the output directory. Users can create a copy of the file called
$subject_id.edit.txt and modify it to change the files that are
converted. This edited file will always overwrite the original file. If
there is a need to revert to original state, please delete this edit.txt
file and rerun the conversion

"""
import argparse
import os
from glob import glob
import subprocess
import sys

import numpy as np

import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.freesurfer as fs
import nipype.pipeline.engine as pe
from nipype.utils.filemanip import save_json, load_json


def get_subjectdcmdir(subjid, dcm_template):
    """Return the TrioTim directory for each subject

    Assumes that this directory is inside the subjects directory
    """
    return glob(dcm_template%subjid)[0]

"""
run a workflow to get at the dicom information
"""

def get_dicom_info(subjs, dcm_template, outputdir):
    """Get the dicom information for each subject

    """
    subjnode = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                       name='subjinfo')
    subjnode.iterables = ('subject_id', subjs)

    infonode = pe.Node(interface=fs.ParseDICOMDir(sortbyrun=True,
                                                  summarize=True),
                       name='dicominfo')

    datasink = pe.Node(interface=nio.DataSink(parameterization=False), name='datasink')
    datasink.inputs.base_directory = outputdir

    infopipe = pe.Workflow(name='extractinfo')
    infopipe.base_dir = os.path.join(outputdir,'workdir')
    infopipe.connect([(subjnode,datasink,[('subject_id','container')]),
                      (subjnode,infonode,[(('subject_id', get_subjectdcmdir, dcm_template),
                                           'dicom_dir')]),
                      (infonode,datasink,[('dicom_info_file','@info')]),
                      ])
    infopipe.run()


def isMoco(dcmfile):
    """Determine if a dicom file is a mocoseries
    """
    cmd = ['mri_probedicom', '--i', dcmfile, '--t', '8', '103e']
    proc  = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    return stdout.strip().startswith('MoCoSeries')

def infotodict(sdir, dicominfofile):
    """Heuristic evaluator for determining which runs belong where
    """
    raise Exception('Please provide your own heuristic file')
    seq = np.genfromtxt(dicominfofile,dtype=object)
    info = dict(bold=[], dwi=[], fieldmap=[], flash=[], mprage=[])
    for s in seq:
        #MPRAGE
        x,y,sl,nt = (int(s[6]),int(s[7]),int(s[8]),int(s[9]))
        if (sl == 176) and (nt == 1) and ('MPRAGE' in s[12]):
            info['mprage'].append(int(s[2]))
        elif  (nt > 100) and (('bold' in s[12]) or ('func' in s[12])):
            if not isMoco(os.path.join(sdir,s[1])):
                info['bold'].append(int(s[2]))
        elif  (sl > 1) and (nt > 25) and ('DIFFUSION' in s[12]):
            info['dwi'].append(int(s[2]))
        elif  s[12].startswith('field_mapping'):
            info['fieldmap'].append(int(s[2]))
        elif  (nt == 8) and s[12].startswith('gre_mgh_multiecho'):
            info['flash'].append(int(s[2]))
        else:
            pass
    return info

def write_cfg(cfgfile, info, sid, tdir, ext='.nii.gz'):
    """Write a config file for unpacksdicomsdir

    If any file that exists is not needed it will remove it from the
    directory. 
    """
    torun = False
    fp = open(cfgfile,'wt')
    for k,runs in info.items():
        if runs:
            field = k
            format = 'nii'
            keepfiles = []
            for r,v in enumerate(runs):
                output = '%s_run%02d_%02d%s' % (sid, r, v, ext)
                outfile = os.path.join(tdir, field, output)
                if not os.path.exists(outfile):
                    torun = True
                    fp.write('%d %s %s %s\n' % (v, field,
                                                format, output))
                keepfiles.append(outfile)
            for file in glob(os.path.join(tdir, field, '*'+ext)):
                if file not in keepfiles:
                    os.remove(file)
            
    fp.close()
    return torun


def convert_dicoms(subjs, dicom_dir_template, outputdir, queue=None, heuristic_func=None,
                   extension = None):
    """Submit conversion jobs to SGE cluster
    """
    if heuristic_func == None:
        heuristic_func = infotodict
    for sid in subjs:
        sdir = dicom_dir_template%sid
        tdir = os.path.join(outputdir, sid)
        infofile =  os.path.join(tdir,'%s.auto.txt' % sid)
        editfile =  os.path.join(tdir,'%s.edit.txt' % sid)
        if os.path.exists(editfile):
            info = load_json(editfile)
        else:
            infofile =  os.path.join(tdir,'%s.auto.txt' % sid)
            info = heuristic_func(sdir, os.path.join(tdir,'dicominfo.txt'))
            save_json(infofile, info)
        cfgfile = os.path.join(tdir,'%s.auto.cfg' % sid)
        if write_cfg(cfgfile, info, sid, tdir, extension):
            convertcmd = ['unpacksdcmdir', '-src', sdir, '-targ', tdir,
                          '-generic', '-cfg', cfgfile, '-skip-moco']
            convertcmd = ' '.join(convertcmd)
            if queue:
                outcmd = 'ezsub.py -n sg-%s -q %s -c \"%s\"'%(sid, queue, convertcmd)
            else:
                outcmd = convertcmd
            os.system(outcmd)


if __name__ == '__main__':
    docstr= '\n'.join((helpdoc,
"""
           Example:

           dicomconvert.py -d rawdata/%s -o . -f heuristic.py -s s1 s2
s3
"""))
    parser = argparse.ArgumentParser(description=docstr)
    parser.add_argument('-d','--dicom_dir_template',
                        dest='dicom_dir_template',
                        required=True,
                        help='location of dicomdir that can be indexed with subject id'
                        )
    parser.add_argument('-s','--subjects',dest='subjs',
                        required=True,
                        type=str, nargs='+',help='list of subjects')
    parser.add_argument('-o','--outdir', dest='outputdir',
                        default=os.getcwd(),
                        help='output directory for conversion')
    parser.add_argument('-f','--heuristic', dest='heuristic_file',
                        help='python script containing heuristic')
    parser.add_argument('-i','--infoonly', dest='infoonly',
                        default=False, action="store_true",
                        help='generate dicominfo file and exit')
    parser.add_argument('-q','--queue',dest='queue',
                        help='SGE queue to use if available')
    parser.add_argument('-x','--ext',dest='ext',default='.nii.gz',
                        help='Output type defaults to .nii.gz')
    args = parser.parse_args()

    
    #dicom_dir_template = '/mindhive/gablab/satra/smoking/rawdata/%s'
    #outputdir = '/mindhive/gablab/satra/smoking/data'
    #heuristic_file = '/mindhive/gablab/satra/smoking/scripts/heuristic.py'
    #queue = 'twocore'
    #subjs = ['rt997', 'rt1']

    heuristic_func = None
    if args.heuristic_file and os.path.exists(args.heuristic_file):
        path, fname = os.path.split(os.path.realpath(args.heuristic_file))
        sys.path.append(path)
        mod = __import__(fname.split('.')[0])
        heuristic_func = mod.infotodict
    get_dicom_info(args.subjs, args.dicom_dir_template,
                   os.path.abspath(args.outputdir))
    if not args.infoonly:
        convert_dicoms(args.subjs, args.dicom_dir_template,
                       os.path.abspath(args.outputdir),
                       heuristic_func=heuristic_func,
                       queue=args.queue,
                       extension = args.ext)

#####################################################
# This file extracts features from stimuli folder   #
# Created by: Yasmin Lipiner Nir                    #
#####################################################

import os

STIMULI_DIR = 'C:\\Users\\yasminln\\ML_Audio_EA\\stimuli'
CONF_DIR = 'C:\\Users\\yasminln\\ML_Audio_EA\\opensmile\\opensmile-2.3.0\\config'
EGEMAPS_CONF_FILE = 'gemaps\\eGeMAPSv01a.conf'
COMPARE_CONF_FILE = 'ComParE_2016.conf'
INTER_CONF_FILE = 'IS12_speaker_trait.conf'
EMO_LARGE_CONF_FILE = 'emo_large.conf'
FRAME_MODE = 'C:\\Users\\yasminln\\ML_Audio_EA\\opensmile\\myconfig\\FrameMode.conf'
OUTPUT_FILE = 'C:\\Users\\yasminln\\ML_Audio_EA\\features\\features_%s.csv'
CONF_FILES = [
    EGEMAPS_CONF_FILE,
    COMPARE_CONF_FILE,
    # INTER_CONF_FILE,
    # EMO_LARGE_CONF_FILE,
]

stimuli_files = os.listdir(STIMULI_DIR)
for conf in CONF_FILES:
    conf_file = os.path.join(CONF_DIR, conf)
    conf_name = conf.split('\\')[-1].split('.')[0]
    print('working on %s config file' % conf_name)
    for stimulus in stimuli_files:
        stimulus_path = os.path.join(STIMULI_DIR, stimulus)
        os.system("SMILExtract_Release -C %s -I %s -O %s -frameModeFunctionalsConf %s -instname \"%s\" "
                  "-appendcsvlld 1 -timestamparff 1" % (conf_file, stimulus_path, OUTPUT_FILE % conf_name, FRAME_MODE,
                                                        stimulus))

print("done extracting features.")

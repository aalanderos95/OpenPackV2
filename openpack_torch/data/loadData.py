from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
from openpack_toolkit import OPENPACK_OPERATIONS
import openpack_toolkit as optk
import os
from pathlib import Path
import shutil
import pandas as pd

def ObtenerDataSet(
    cfg: DictConfig,
    nameActivitiesFile: str,
    nameDataset: str = "default",
    classes: optk.ActSet = OPENPACK_OPERATIONS,
):
    data_dir = cfg.datadir
    data_dir_freq = cfg.data_dir_freq
    #create training dir
    training_dir = os.path.join(data_dir,"training")
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)


    #create dataset training
    trainDir = os.path.join(training_dir, nameDataset)
    if not os.path.isdir(trainDir):
        os.mkdir(trainDir)

    
    split = cfg.dataset.split


    #TRAIN DATA
    pathTrain  = f"{data_dir_freq}/{cfg.sampling}/train/"
    
    for dirpath, dirnames, filenames in os.walk(pathTrain):
        for filename in filenames:
            if ".jpg" in filename.lower():
                if not os.path.exists(os.path.join(trainDir, filename)):
                    shutil.copy(os.path.join(dirpath, filename),trainDir)
    
    labelsTrain = f"{data_dir_freq}/{cfg.sampling}/train/{nameActivitiesFile}"

    #VALIDATION DATA
    labelsVal = dict()
    valDir = dict()
    for user, session in split.val:
        pathUser = f"{user}-{session}"
        labelsVal[pathUser] = f"{data_dir_freq}/{cfg.sampling}/val/{pathUser}/{nameActivitiesFile}"
        valDir[pathUser] = f"{data_dir_freq}/{cfg.sampling}/val/{pathUser}/img"
    
    
    
    #TEST DATA
    labelsTest = dict()
    testDir = dict()
    for user, session in split.test:
        pathUser = f"{user}-{session}"
        labelsTest[pathUser] = f"{data_dir_freq}/{cfg.sampling}/test/{pathUser}/{nameActivitiesFile}"
        testDir[pathUser] = f"{data_dir_freq}/{cfg.sampling}/test/{pathUser}/img"
    
    
    
    #SUBMISSION DATA
    labelsSubmission = dict()
    submissionDir = dict()
    for user, session in split.submission:
        pathUser = f"{user}-{session}"
        labelsSubmission[pathUser] = f"{data_dir_freq}/{cfg.sampling}/submission/{pathUser}/{nameActivitiesFile}"
        submissionDir[pathUser] = f"{data_dir_freq}/{cfg.sampling}/submission/{pathUser}/img"
    
    
    return trainDir, labelsTrain, valDir, labelsVal, testDir, labelsTest, submissionDir, labelsSubmission;


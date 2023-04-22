"""Dataset Class for OpenPack dataset.
"""
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openpack_toolkit as optk
import torch
from torchvision import transforms

from omegaconf import DictConfig, open_dict
from openpack_toolkit import OPENPACK_OPERATIONS
import numpy as np
import pandas as pd
import time

from einops import rearrange
import matplotlib.image as img

from pykalman import KalmanFilter
from .dataloader import (load_imu_all,load_imu_new,load_imu, loadImage, load_e4acc, load_imu_ant)
import os
logger = getLogger(__name__)


class OpenPackImu(torch.utils.data.Dataset):
    """Dataset class for IMU data.

    Attributes:
        data (List[Dict]): each sequence is stored in dict. The dict has 5 keys (i.e.,
            user, session, data, label(=class index), unixtime). data is a np.ndarray with
            shape = ``(N, channel(=acc_x, acc_y, ...), window, 1)``.
        index (Tuple[Dict]): sample index. A dict in this tuple as 3 property.
            ``seq`` = sequence index, ``sqg`` = segment index which is a sequential number
            within the single sequence. ``pos`` = sample index of the start of this segment.
        classes (optk.ActSet): list of activity classes.
        window (int): sliding window size.
        debug (bool): If True, enable debug mode. Default to False.
        submission (bool): Set True when you make submission file. Annotation data will not be
            loaded and dummy data will be generated. Default to False.

    Todo:
        * Make a minimum copy of cfg (DictConfig) before using in ``load_dataset()``.
        * Add method for parameter validation (i.e., assert).
    """
    data: List[Dict] = None
    index: Tuple[Dict] = None

    def __init__(
            self,
            cfg: DictConfig,
            user_session_list: Tuple[Tuple[int, int], ...],
            classes: optk.ActSet = OPENPACK_OPERATIONS,
            window: int = 30 * 60,
            submission: bool = False,
            debug: bool = False,
    ) -> None:
        """Initialize OpenPackImu dataset class.

        Args:
            cfg (DictConfig): instance of ``optk.configs.OpenPackConfig``. path, dataset, and
                annotation attributes must be initialized.
            user_session (Tuple[Tuple[int, int], ...]): the list of pairs of user ID and session ID
                to be included.
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
            window (int, optional): window size [steps]. Defaults to 30*60 [s].
            submission (bool, optional): Set True when you want to load test data for submission.
                If True, the annotation data will no be replaced by dummy data. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.
        """
        super().__init__()
        self.classes = classes
        self.window = window
        self.submission = submission
        self.debug = debug

        self.load_dataset(
            cfg,
            user_session_list,
            window,
            submission=submission)

        self.preprocessing()

    def load_dataset(
        self,
        cfg: DictConfig,
        user_session_list: Tuple[Tuple[int, int], ...],
        window: int = None,
        submission: bool = False,
    ) -> None:
        """Called in ``__init__()`` and load required data.

        Args:
            user_session (Tuple[Tuple[str, str], ...]): _description_
            window (int, optional): _description_. Defaults to None.
            submission (bool, optional): _description_. Defaults to False.
        """
        data, index = [], []
        for seq_idx, (user, session) in enumerate(user_session_list):
            with open_dict(cfg):
                cfg.user = {"name": user}
                cfg.session = session

            paths_imu = []
            for device in cfg.dataset.stream.devices:
                with open_dict(cfg):
                    cfg.device = device

                path = Path(
                    cfg.dataset.stream.path.dir,
                    cfg.dataset.stream.path.fname
                )
                paths_imu.append(path)

            ts_sess, x_sess = optk.data.load_imu(
                paths_imu,
                use_acc=True)

            if submission:
                # For set dummy data.
                label = np.zeros((len(ts_sess),), dtype=np.int64)
            else:
                path = Path(
                    cfg.dataset.annotation.path.dir,
                    cfg.dataset.annotation.path.fname
                )
                df_label = optk.data.load_and_resample_operation_labels(
                    path, ts_sess, classes=self.classes)
                label = df_label["act_idx"].values

            data.append({
                "user": user,
                "session": session,
                "data": x_sess,
                "label": label,
                "unixtime": ts_sess,
            })

            seq_len = ts_sess.shape[0]
            index += [dict(seq=seq_idx, seg=seg_idx, pos=pos)
                      for seg_idx, pos in enumerate(range(0, seq_len, window))]
        self.data = data
        self.index = tuple(index)


    def preprocessing(self) -> None:
        """This method is called after ``load_dataset()`` and apply preprocessing to loaded data.
        """
        logger.warning("No preprocessing is applied.")

    @property
    def num_classes(self) -> int:
        """Returns the number of classes

        Returns:
            int
        """
        return len(self.classes)

    def __str__(self) -> str:
        s = (
            "OpenPackImu("
            f"index={len(self.index)}, "
            f"num_sequence={len(self.data)}, "
            f"submission={self.submission}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict:
        seq_idx, seg_idx = self.index[index]["seq"], self.index[index]["seg"]
        seq_dict = self.data[seq_idx]
        seq_len = seq_dict["data"].shape[1]

        head = seg_idx * self.window
        tail = (seg_idx + 1) * self.window
        if tail >= seq_len:
            pad_tail = tail - seq_len
            tail = seq_len
        else:
            pad_tail = 0
        assert (
            head >= 0) and (
            tail > head) and (
            tail <= seq_len), f"head={head}, tail={tail}"

        x = seq_dict["data"][:, head:tail, np.newaxis]
        t = seq_dict["label"][head:tail]
        ts = seq_dict["unixtime"][head:tail]

        if pad_tail > 0:
            x = np.pad(x, [(0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            t = np.pad(t, [(0, pad_tail)], mode="constant",
                       constant_values=self.classes.get_ignore_class_index())
            ts = np.pad(ts, [(0, pad_tail)],
                        mode="constant", constant_values=ts[-1])

        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        ts = torch.from_numpy(ts)
        return {"x": x, "t": t, "ts": ts}

class OpenPackImuAll(torch.utils.data.Dataset):
    """Dataset class for IMU data.

    Attributes:
        data (List[Dict]): each sequence is stored in dict. The dict has 5 keys (i.e.,
            user, session, data, label(=class index), unixtime). data is a np.ndarray with
            shape = ``(N, channel(=acc_x, acc_y, ...), window, 1)``.
        index (Tuple[Dict]): sample index. A dict in this tuple as 3 property.
            ``seq`` = sequence index, ``sqg`` = segment index which is a sequential number
            within the single sequence. ``pos`` = sample index of the start of this segment.
        classes (optk.ActSet): list of activity classes.
        window (int): sliding window size.
        debug (bool): If True, enable debug mode. Default to False.
        submission (bool): Set True when you make submission file. Annotation data will not be
            loaded and dummy data will be generated. Default to False.

    Todo:
        * Make a minimum copy of cfg (DictConfig) before using in ``load_dataset()``.
        * Add method for parameter validation (i.e., assert).
    """
    data: List[Dict] = None
    index: Tuple[Dict] = None

    def __init__(
            self,
            cfg: DictConfig,
            user_session_list: Tuple[Tuple[int, int], ...],
            classes: optk.ActSet = OPENPACK_OPERATIONS,
            window: int = 30 * 60,
            submission: bool = False,
            debug: bool = False,
    ) -> None:
        """Initialize OpenPackImu dataset class.

        Args:
            cfg (DictConfig): instance of ``optk.configs.OpenPackConfig``. path, dataset, and
                annotation attributes must be initialized.
            user_session (Tuple[Tuple[int, int], ...]): the list of pairs of user ID and session ID
                to be included.
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
            window (int, optional): window size [steps]. Defaults to 30*60 [s].
            submission (bool, optional): Set True when you want to load test data for submission.
                If True, the annotation data will no be replaced by dummy data. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.
        """
        super().__init__()
        self.classes = classes
        self.window = window
        self.submission = submission
        self.debug = debug
        self.resData = cfg.resData
        self.time_step = cfg.time_step
        self.cfg = cfg
    
        self.load_dataset(
            cfg,
            user_session_list,
            window,
            submission=submission)

        self.preprocessing()

    def load_dataset(
        self,
        cfg: DictConfig,
        user_session_list: Tuple[Tuple[int, int], ...],
        window: int = None,
        submission: bool = False,
    ) -> None:
        """Called in ``__init__()`` and load required data.

        Args:
            user_session (Tuple[Tuple[str, str], ...]): _description_
            window (int, optional): _description_. Defaults to None.
            submission (bool, optional): _description_. Defaults to False.
        """
        data, index = [], []
        inicio = time.time()

        for seq_idx, (user, session) in enumerate(user_session_list):
            x_sAll, ts_sAll, nameData = [], [], []
            with open_dict(cfg):
                cfg.user = {"name": user}
                cfg.session = session

            #IMU [0]
            paths_imu = []
            for device in cfg.dataset.stream.devices_imu:
                with open_dict(cfg):
                    cfg.device = device

                path = Path(
                    cfg.dataset.stream.path_imu.dir,
                    cfg.dataset.stream.path_imu.fname
                )
                paths_imu.append(path)

            ts_sess, x_sess = optk.data.load_imu(
                paths_imu,
                use_acc=cfg.dataset.stream.acc,
                use_gyro=cfg.dataset.stream.gyro,
                use_quat=cfg.dataset.stream.quat,
                th=cfg.dataset.stream.frame_rate_imu)

            x_sAll.append(x_sess)
            ts_sAll.append(ts_sess)
            nameData.append("IMU")
            #E4 ACC [1]
            paths_e4acc = []
            for device in cfg.dataset.stream.devices_e4acc:
                with open_dict(cfg):
                    cfg.device = device
                path = Path(
                    cfg.dataset.stream.path_e4acc.dir,
                    cfg.dataset.stream.path_e4acc.fname
                )
                paths_e4acc.append(path)

            x_sess_e4acc, ts_sess_e4acc = load_e4acc(paths_e4acc)

            x_sAll.append(x_sess_e4acc)
            ts_sAll.append(ts_sess_e4acc)
            nameData.append("e4acc")
            #KEYPOINTS [2]
            pathKeypoints = Path(
                cfg.dataset.stream.path_keypoint.dir,
                cfg.dataset.stream.path_keypoint.fname,
            )
            
            ts_sess_keypoint, x_sess_keypoints = optk.data.load_keypoints(pathKeypoints)
            x_sess_keypoints = x_sess_keypoints[:(x_sess_keypoints.shape[0] - 1)]  # Remove prediction score.
            x_sess_keypoints = x_sess_keypoints.transpose(1, 0, 2)
            
           
            x_sAll.append(x_sess_keypoints)
            ts_sAll.append(ts_sess_keypoint)
            nameData.append("keypoints")

           
            x_sAll, ts_sAll = self.remuestreo_padding(x_sAll, ts_sAll, nameData)


            #HT [3]
            pathht = Path(
                cfg.dataset.stream.path_ht.dir,
                cfg.dataset.stream.path_ht.fname,
            )
            x_sess_ht = optk.data.load_and_resample_scan_log(pathht, ts_sAll[0])
            
            x_sAll.append(x_sess_ht)

            #Printer [4]

            path_printer = Path(
                cfg.dataset.stream.path_printer.dir,
                cfg.dataset.stream.path_printer.fname,
            )

            x_sess_printer = optk.data.load_and_resample_scan_log(path_printer, ts_sAll[0])

            x_sAll.append(x_sess_printer)



            if submission:
                # For set dummy data.
                label = np.zeros((len(ts_sAll[0]),), dtype=np.int64)
            else:
                path = Path(
                    cfg.dataset.annotation.path.dir,
                    cfg.dataset.annotation.path.fname
                )
                dflabel = optk.data.load_and_resample_operation_labels(
                    path, ts_sAll[0], classes=self.classes)
                
                label = dflabel["act_idx"].values    

            data.append({
                "user": user,
                "session": session,
                "dataimu": x_sAll[0],
                "datae4acc": x_sAll[1],
                "datakeypoints": x_sAll[2],
                "dataht": x_sAll[3],
                "dataprinter": x_sAll[4],
                "label": label,
                "unixtime": ts_sAll[0],
            })

            seq_lenimu = ts_sAll[0].shape[0]
            
            index += [dict(seq=seq_idx, seg=seg_idx, pos=pos)
                      for seg_idx, pos in enumerate(range(0, seq_lenimu, self.window))]

        fin = time.time()
        logger.info(f"Tiempo de carga: {(fin-inicio)}!") 
        self.data = data
        self.index = tuple(index)
        indice = 0
        if submission:
            for seq_dict in self.data:
                if seq_dict["label"].shape[0] != seq_dict["datae4acc"].shape[1]:
                    print(f"NO COINSIDE EL INDICE {indice}")
                    print(seq_dict["label"].shape)
                    print(seq_dict["datae4acc"].shape)
                    print(seq_dict["user"])
                    print(seq_dict["session"])
           
        
    def min_maxLabels(self, xslist, tslist):
        xsNList, tsNList = [], []
        mints_list, maxts_list = [], []
        for ts in tslist:
            mints_list.append(ts[0])
            maxts_list.append(ts[-1])

        mints = max(np.array(mints_list))
        mints = mints - (mints % 1000)
        maxts = min(np.array(maxts_list))
        maxts = maxts - (maxts % 1000)
        
        for i, (ts, x) in enumerate(zip(tslist, xslist)):
            use_index = np.logical_and(ts >= mints, ts <= maxts)
            ts = ts[use_index]
            xN = []
            if(i == 2):
                xN.append(x[use_index])
            else:
                for xn in x:
                    xN.append(xn[use_index])
                        
            xN = np.array(xN)        

            xsNList.append(xN)
            tsNList.append(ts)
        
        labels = np.arange(mints, maxts+1000, 1000)
        return labels, xsNList, tsNList


    def remuestreo_padding(self, xslist, tslist, nameData):
            xsNList, tsNList = [], []
            mints_list, maxts_list, noneList = [], [], []
            for ts in tslist:
                if type(ts) == np.ndarray:
                    mints_list.append(ts[0])
                    maxts_list.append(ts[-1])
                    noneList.append(False)
                else:                  
                    assert ts == None
                    mints_list.append(10000000)
                    maxts_list.append(0)
                    noneList.append(True)
            
            indicesValidos =  [int(i) for i, x in enumerate(noneList) if x == False]

            mints = max(np.array(mints_list)[indicesValidos])
            mints = mints - (mints % 1000)
            maxts = min(np.array(maxts_list)[indicesValidos])
            maxts = maxts - (maxts % 1000)
            
            for i, (ts, x) in enumerate(zip(tslist, xslist)):                
                if noneList[i] == True:
                    if nameData[i] == "e4acc":
                        ts = tsNList[0]
                        x = np.zeros((len(ts), self.cfg.channelse4acc))
                        x = x.transpose(1,0)
                        xsNList.append(x)    
                        tsNList.append(ts)
                else:
                    x, ts = self._remuestrear(x, ts, nameData[i])             

                    use_index = np.logical_and(ts >= mints, ts <= maxts)
                    ts = ts[use_index]
                    xN = []
                    if nameData[i] == "keypoints":   
                        xsNList.append(x[:,:,use_index])
                    else:
                        for xn in x:
                            xN.append(xn[use_index])
                        xN = np.array(xN) 
                        xsNList.append(xN)    
                    
                    tsNList.append(ts)
        
            #xsNList, tsNList = self.padding(xsNList, tsNList, nameData)
            return  xsNList, tsNList
    def padding(self, xs, ts, nameData):
        xLen = []
        xLen.append(xs[0].shape[1])
        xLen.append(xs[1].shape[1])
        xLen.append(xs[2].shape[2])
        
        tMin = max(len(xLen), min(xLen))
        for i, (tsI, x) in enumerate(zip(ts, xs)):
            ts[i] = tsI[:tMin]

            if nameData[i] == "keypoints":
                xs[i] = x[:,:tMin]
            else:     
                xs[i] = x[:,:tMin]
        
        return xs, ts

    def _remuestrear(self, xs, ts, nameData):
        nResData = str(int(1000 / self.resData)) + 'L' 
        df_ts = pd.DataFrame()
        df_ts["unixtime"] = pd.Series(ts)
        pUnix = False
        if nameData == "keypoints":
            xsFinal = []
            xs = xs.transpose(1,2,0)
            
            arrNodes = []
            for x in range(len(xs)):
                df_xs = pd.DataFrame()                            
                for node in range(len(xs[x])):
                    df_xs[node] = pd.Series(xs[x][node])
                
                df_concat = pd.concat([df_xs, df_ts], axis=1)
                df_concat["Resample"] = df_concat['unixtime'].astype('datetime64[ms]')
                df_concat = df_concat.set_index('Resample')
                
                df_concat = df_concat.reset_index().groupby(
                    pd.Grouper(
                        freq=nResData,
                        key='Resample')).mean().interpolate(
                    method='linear',
                    limit_direction='both',
                    axis=0)
                df_concat['unixtime'] = df_concat.index.to_series().apply(
                            lambda x: np.int64(str(pd.Timestamp(x).value)[0:13]))
                                
                if pUnix == False:
                    tsFinal = df_concat['unixtime'].values
                    pUnix = True
                
                df_concat = df_concat.drop(df_concat.columns[-1], axis=1)    
                arrNodes.append(df_concat.values.T)
            arrNodes = np.array(arrNodes)
           
            xsFinal.append(arrNodes)
            xsFinal = np.array(xsFinal).squeeze()
            xs = xsFinal
            ts = tsFinal
        else:
            df_xs = pd.DataFrame()
           
            for i in range(len(xs)):
                df_xs[i] = pd.Series(xs[i])

            df_concat = pd.concat([df_xs, df_ts], axis=1)
            df_concat["Resample"] = df_concat['unixtime'].astype('datetime64[ms]')
            df_concat = df_concat.set_index('Resample')
            
            df_concat = df_concat.reset_index().groupby(
                pd.Grouper(
                    freq=nResData,
                    key='Resample')).mean().interpolate(
                method='linear',
                limit_direction='both',
                axis=0)
            df_concat['unixtime'] = df_concat.index.to_series().apply(
                        lambda x: np.int64(str(pd.Timestamp(x).value)[0:13]))
            
            ts = df_concat['unixtime'].values
            df_concat = df_concat.drop(df_concat.columns[-1], axis=1)        
            xs = df_concat.values.T
        
        return xs, ts
        
    def preprocessing(self) -> None:
        #Normalization
        if self.cfg.normalization:
            
            for seq_dict in self.data:
                """IMU DATA"""
                xIMU = seq_dict.get("dataimu")
                for i in range(len(xIMU)):
                    min_value = np.max(xIMU[i])
                    max_value = np.min(xIMU[i])
                    xIMU[i] = (xIMU[i] - min_value) / (max_value - min_value)
                    #xIMU[i] = (xIMU[i] - 0.5) / 0.5            
                seq_dict["dataimu"] = xIMU
                xe4Acc = seq_dict.get("datae4acc")
                for i in range(len(xe4Acc)):
                    min_value = np.max(xe4Acc[i])
                    max_value = np.min(xe4Acc[i])
                    xe4Acc[i] = (xe4Acc[i] - min_value) / (max_value - min_value)
                seq_dict["datae4acc"] = xe4Acc
                
                xkeypoints = seq_dict.get("datakeypoints")
                for i in range(len(xkeypoints)):
                    for j in range(len(xkeypoints[i])):
                        min_value = np.max(xkeypoints[i][j])
                        max_value = np.min(xkeypoints[i][j])
                        xkeypoints[i][j] = (xkeypoints[i][j] - min_value) / (max_value - min_value)
                seq_dict["datakeypoints"] = xkeypoints

    @property
    def num_classes(self) -> int:
        """Returns the number of classes

        Returns:
            int
        """
        return len(self.classes)

    def __str__(self) -> str:
        s = (
            "OpenPackImu("
            f"index={len(self.index)}, "
            f"num_sequence={len(self.data)}, "
            f"submission={self.submission}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return self

    def segmentar(self, data, unixtime, start_ts, end_ts, fs):
        _start_ind = np.where(unixtime >= start_ts)[0]
        #TOMAR LOS ULTIMOS SI EXCEDE EL TAMAÑO
        if len(_start_ind) == 0:
            start_ind = len(unixtime) - 2
            end_ind = len(unixtime) - 1
        else:
            start_ind = _start_ind[0]
            _end_ind = np.where(unixtime <= end_ts)[0]
            if len(_end_ind) == 0:
                end_ind = None
            else:
                end_ind = _end_ind[-1]
                if start_ind == end_ind:
                    end_ind = None
        
        print("PRIMERO",data.shape, unixtime.shape)

        data = data[:, start_ind:end_ind, np.newaxis]
        unixtime = unixtime[start_ind:end_ind]

        print("DESPUES", data.shape, unixtime.shape)
        exit()
        data = self._segment_and_padding(data, unixtime, fs)
        return data, unixtime
    
    def _segment_and_padding(self, data, ts, fs):
        data_segment = []
        num_data = int(self.time_step / 1000 * fs)
        if num_data == 0:  # for label's unixtime
            num_data = 1
        for i in range(self.resData):
            _segment_start = np.where(ts >= ts[0] + (i * self.time_step))
            if len(_segment_start[0]) == 0:
                segment_start = ts.shape[0] - 1
            else:
                segment_start = _segment_start[0][0]
            segment_end = segment_start + num_data
            if segment_end >= len(data):
                # padding with last data
                last_data = data[-1]
                len_repeat = num_data - len(data[segment_start:])
                # if len_repeat < 0:
                #     segment_end = segment_start + num_data
                #     data_segment.append(data[segment_start:segment_end])
                # else:
                repeat_data = np.repeat([last_data], len_repeat, axis=0)
                data_segment.append(np.concatenate([data[segment_start:], repeat_data]))
                for j in range(i + 1, self.resData):
                    len_repeat = num_data
                    repeat_data = np.repeat([last_data], len_repeat, axis=0)
                    data_segment.append(repeat_data)
                break
            else:
                assert segment_end - segment_end <= fs
            data_segment.append(data[segment_start:segment_end])

        data_segment = np.array(data_segment)
        assert len(data_segment) == self.resData
        assert data_segment.shape[1] == num_data

        return np.array(data_segment)
    def __getitem__(self, index: int) -> Dict:        
        seq_idx, seg_idx = self.index[index]["seq"], self.index[index]["seg"]
        seq_data = self.data[seq_idx]
        seq_len = seq_data["dataimu"].shape[1]

        head = seg_idx * self.window
        tail = (seg_idx + 1) * self.window
        if tail >= seq_len:
            pad_tail = tail - seq_len
            tail = seq_len
        else:
            pad_tail = 0
        assert (
            head >= 0) and (
            tail > head) and (
            tail <= seq_len), f"head={head}, tail={tail}"
        label = seq_data["label"][head:tail]
        ts = seq_data["unixtime"][head:tail]


        #IMU
        imu = seq_data["dataimu"][:, head:tail, np.newaxis]
        
        #E4ACC
        e4acc = seq_data["datae4acc"][:, head:tail, np.newaxis]

        #KEYPOINTS
        keypoints = seq_data["datakeypoints"][:,:, head:tail, np.newaxis]
        
        #HT
        ht = seq_data["dataht"][head:tail,np.newaxis]

        #PRINTER
        printer = seq_data["dataprinter"][head:tail,np.newaxis]


        if pad_tail > 0:
            imu = np.pad(imu, [(0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            
            e4acc = np.pad(e4acc, [(0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            
            ht = np.pad(ht, [(0, pad_tail), (0,0)],
                       mode="constant", constant_values=0)
            printer = np.pad(printer, [(0, pad_tail),(0,0)],
                       mode="constant", constant_values=0)
           
            keypoints = np.pad(keypoints, [(0, 0), (0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            label = np.pad(label, [(0, pad_tail)], mode="constant",
                       constant_values=self.classes.get_ignore_class_index())
            ts = np.pad(ts, [(0, pad_tail)],
                        mode="constant", constant_values=ts[-1])
          
        x_imu = torch.from_numpy(imu)
        label = torch.from_numpy(label)
        ts = torch.from_numpy(ts)
        x_e4acc = torch.from_numpy(e4acc)
        x_keypoints = torch.from_numpy(keypoints)
        ht = torch.from_numpy(ht)
        printer = torch.from_numpy(printer)

        return {
            "x_imu": x_imu, 
            "label": label, 
            "unixtime": ts,
            "x_e4acc": x_e4acc,
            "x_keypoints": x_keypoints,
            "x_ht": ht,
            "x_printer": printer,
            }



class OpenPackImuMulti(torch.utils.data.Dataset):
    """Dataset class for IMU data.

    Attributes:
        data (List[Dict]): each sequence is stored in dict. The dict has 5 keys (i.e.,
            user, session, data, label(=class index), unixtime). data is a np.ndarray with
            shape = ``(N, channel(=acc_x, acc_y, ...), window, 1)``.
        index (Tuple[Dict]): sample index. A dict in this tuple as 3 property.
            ``seq`` = sequence index, ``sqg`` = segment index which is a sequential number
            within the single sequence. ``pos`` = sample index of the start of this segment.
        classes (optk.ActSet): list of activity classes.
        window (int): sliding window size.
        debug (bool): If True, enable debug mode. Default to False.
        submission (bool): Set True when you make submission file. Annotation data will not be
            loaded and dummy data will be generated. Default to False.

    Todo:
        * Make a minimum copy of cfg (DictConfig) before using in ``load_dataset()``.
        * Add method for parameter validation (i.e., assert).
    """
    data: List[Dict] = None
    index: Tuple[Dict] = None
    sampling: int = None

    def __init__(
            self,
            cfg: DictConfig,
            user_session_list: Tuple[Tuple[int, int], ...],
            classes: optk.ActSet = OPENPACK_OPERATIONS,
            window: int = 30 * 60,
            submission: bool = False,
            debug: bool = False,
    ) -> None:
        """Initialize OpenPackImu dataset class.

        Args:
            cfg (DictConfig): instance of ``optk.configs.OpenPackConfig``. path, dataset, and
                annotation attributes must be initialized.
            user_session (Tuple[Tuple[int, int], ...]): the list of pairs of user ID and session ID
                to be included.
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
            window (int, optional): window size [steps]. Defaults to 30*60 [s].
            submission (bool, optional): Set True when you want to load test data for submission.
                If True, the annotation data will no be replaced by dummy data. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.
        """
        super().__init__()
        self.classes = classes
        self.window = window
        self.submission = submission
        self.debug = debug
        self.sampling = cfg.sampling
        self.normalization = cfg.normalization
        self.standardNormalization = cfg.standardNormalization
        self.kalman = cfg.kalman

        self.load_dataset(
            cfg,
            user_session_list,
            window,
            submission=submission)

        self.preprocessing()

    def load_dataset(
        self,
        cfg: DictConfig,
        user_session_list: Tuple[Tuple[int, int], ...],
        window: int = None,
        submission: bool = False,
    ) -> None:
        """Called in ``__init__()`` and load required data.

        Args:
            user_session (Tuple[Tuple[str, str], ...]): _description_
            window (int, optional): _description_. Defaults to None.
            submission (bool, optional): _description_. Defaults to False.
        """
        #Validar si existe annotation antes de obtener datos
       
        data, index = [], []
        inicio = time.time()
        for seq_idx, (user, session) in enumerate(user_session_list):
            with open_dict(cfg):
                cfg.user = {"name": user}
                cfg.session = session

            pathAnnotation = Path(
                cfg.dataset.annotation.path.dir,
                cfg.dataset.annotation.path.fname
            )
            if(os.path.exists(pathAnnotation) or submission):
                paths_imu = []
                channels = []
                hz = []
                cont = 0
                pathsWOSession = []
                for stream in cfg.dataset.stream:
                    for device in stream.devices:
                        with open_dict(cfg):
                            cfg.device = device

                        path = Path(
                            stream.path.dir,
                            stream.path.fname
                        )
                        pathsWOSession.append(stream.path.dir)
                        paths_imu.append(path)
                        hz.append(stream.frame_rate)
                        if "atr" in str(path):
                            if stream.acc in (None, True):
                                if channels == [] or len(channels) < (cont + 1):
                                    channels.append(["acc_x", "acc_y", "acc_z"])
                                else:
                                    channels[cont] += ["acc_x", "acc_y", "acc_z"]
                            if stream.gyro in (None, True):
                                if channels == [] or len(channels) < (cont + 1):
                                    channels.append(["gyro_x", "gyro_y", "gyro_z"])
                                else:
                                    channels[cont] += ["gyro_x",
                                                    "gyro_y", "gyro_z"]
                            if stream.quat in (None, True):
                                if channels == [] or len(channels) < (cont + 1):
                                    channels.append(
                                        ["quat_w", "quat_x", "quat_y", "quat_z"])
                                else:
                                    channels[cont] += ["quat_w",
                                                    "quat_x", "quat_y", "quat_z"]
                        elif "acc" in str(path):
                            if channels == [] or len(channels) < (cont + 1):
                                channels.append(["acc_x", "acc_y", "acc_z"])
                            else:
                                channels[cont] += ["acc_x", "acc_y", "acc_z"]
                        elif "eda" in str(path):
                            if channels == [] or len(channels) < (cont + 1):
                                channels.append(["eda"])
                            else:
                                channels[cont] += ["eda"]
                        elif "bvp" in str(path):
                            if channels == [] or len(channels) < (cont + 1):
                                channels.append(["bvp"])
                            else:
                                channels[cont] += ["bvp"]
                        elif "temp" in str(path):
                            if channels == [] or len(channels) < (cont + 1):
                                channels.append(["temp"])
                            else:
                                channels[cont] += ["temp"]
                        cont = cont + 1
                ts_sess, x_sess  =  load_imu_new(
                    paths_imu,
                    pathsWOSession,
                    channels,
                    self.sampling,
                    hz,
                    cfg.statistics)
                    
                if submission:
                    # For set dummy data.
                    label = np.zeros((len(ts_sess),), dtype=np.int64)
                else:
                    path = Path(
                        cfg.dataset.annotation.path.dir,
                        cfg.dataset.annotation.path.fname
                    )
                    df_label = optk.data.load_and_resample_operation_labels(
                        path, ts_sess, classes=self.classes)
                    label = df_label["act_idx"].values

                data.append({
                    "user": user,
                    "session": session,
                    "data": x_sess,
                    "label": label,
                    "unixtime": ts_sess,
                })
                
                
                seq_len = ts_sess.shape[0]
                index += [dict(seq=seq_idx, seg=seg_idx, pos=pos)
                            for seg_idx, pos in enumerate(range(0, seq_len, window))]
        
        fin = time.time()
        logger.info(f"Tiempo de Ejecución: {(fin-inicio)}!") 
        self.data = data
        self.index = tuple(index)

    def preprocessing(self) -> None:
        if self.normalization:
            for seq_dict in self.data:
                x = seq_dict.get("data")
                
                for xVal in range(len(x)):
                    df = pd.DataFrame(data = x[xVal], columns = ["value"])
                    df = df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
                    
                    #df["value"] = (df["value"] - df["value"].min())/(df["value"].max()-df["value"].min())
                    x[xVal] = df.values.T
                seq_dict["data"] = x
        elif self.standardNormalization:
            for seq_dict in self.data:
                x = seq_dict.get("data")
                x = torch.from_numpy(x)
                x = StandardScaler().fit_transform(x)
                seq_dict["data"] = x.numpy()

        if self.kalman:
            #KALMAN FILTER
            inicio = time.time()

            observation_covariance = .0015

            contador = 0
            for seq_dict in self.data:
                x = seq_dict.get("data")
                
                for x_ in range(len(x)):
                    x_kalman = self.Kalman1D(x[x_],observation_covariance)
                    contador = contador +1
                seq_dict["data"] = x

                
    
            fin = time.time()
            logger.info(f"TIEMPO KALMAN PARA {contador}: {(fin-inicio)}!") 
    def Kalman1D(self, observations,damping=1):
    # To return the smoothed time series data
        observation_covariance = damping
        initial_value_guess = observations[0]
        transition_matrix = 1
        transition_covariance = 0.1
        initial_value_guess
        kf = KalmanFilter(
                initial_state_mean=initial_value_guess,
                initial_state_covariance=observation_covariance,
                observation_covariance=observation_covariance,
                transition_covariance=transition_covariance,
                transition_matrices=transition_matrix
            )
        pred_state, state_cov = kf.smooth(observations)
        return pred_state


    @property
    def num_classes(self) -> int:
        """Returns the number of classes

        Returns:
            int
        """
        return len(self.classes)

    def __str__(self) -> str:
        s = (
            "OpenPackImuMulti("
            f"index={len(self.index)}, "
            f"num_sequence={len(self.data)}, "
            f"submission={self.submission}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict:
        seq_idx, seg_idx = self.index[index]["seq"], self.index[index]["seg"]
        seq_dict = self.data[seq_idx]
    
        seq_len = seq_dict["data"].shape[1]

        head = seg_idx * self.window
        tail = (seg_idx + 1) * self.window
        if tail >= seq_len:
            pad_tail = tail - seq_len
            tail = seq_len
        else:
            pad_tail = 0
        assert (
            head >= 0) and (
            tail > head) and (
            tail <= seq_len), f"head={head}, tail={tail}"

        x = seq_dict["data"][:, head:tail, np.newaxis]
        t = seq_dict["label"][head:tail]
        ts = seq_dict["unixtime"][head:tail]

        if pad_tail > 0:
            x = np.pad(x, [(0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            t = np.pad(t, [(0, pad_tail)], mode="constant",
                       constant_values=self.classes.get_ignore_class_index())
            ts = np.pad(ts, [(0, pad_tail)],
                        mode="constant", constant_values=ts[-1])

        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        ts = torch.from_numpy(ts)
        return {"x": x, "t": t, "ts": ts}

# -----------------------------------------------------------------------------

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

class OpenPackKeypoint(torch.utils.data.Dataset):
    """Dataset Class for Keypoint Data.

    Attributes:
        data (List[Dict]): shape = (N, 3, FRAMES, VERTEX)
        index (Tuple[Dict]): sample index. A dict in this tuple as 3 property.
            ``seq`` = sequence index, ``sqg`` = segment index which is a sequential number
            within the single sequence. ``pos`` = sample index of the start of this segment.
        classes (Tuple[ActClass]): list of activity classes.
        window (int): window size (=the number of frames per sample)
        device (torch.device): -
        dtype (Tuple[torch.dtype,torch.dtype]): -
    """
    data: List[Dict] = None
    index: Tuple[Dict] = None

    def __init__(
            self,
            cfg: DictConfig,
            user_session: Tuple[Tuple[int, int], ...],
            classes: optk.ActSet = OPENPACK_OPERATIONS,
            window: int = 15 * 60,
            submission: bool = False,
            debug: bool = False,
    ) -> None:
        """Initialize OpenPackKyepoint dataset class.

        Args:
            cfg (DictConfig): instance of ``optk.configs.OpenPackConfig``. path, dataset, and
                annotation attributes must be initialized.
            user_session (Tuple[Tuple[int, int], ...]): _description_
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
            window (int, optional): window size. Defaults to 15*60 [frames].
            submission (bool, optional): _description_. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.
        """
        super().__init__()
        self.window = window
        self.classes = classes
        self.submission = submission
        self.debug = debug

        self.load_dataset(
            cfg,
            user_session,
            submission=submission)

        self.preprocessing()

    def load_dataset(
        self,
        cfg: DictConfig,
        user_session: Tuple[Tuple[int, int], ...],
        submission: bool = False,
    ):
        data, index = [], []
        for seq_idx, (user, session) in enumerate(user_session):
            with open_dict(cfg):
                cfg.user = {"name": user}
                cfg.session = session

            path = Path(
                cfg.dataset.stream.path.dir,
                cfg.dataset.stream.path.fname,
            )
            ts_sess, x_sess = optk.data.load_keypoints(path)
            x_sess = x_sess[:(x_sess.shape[0] - 1)]  # Remove prediction score.

            if submission:
                # For set dummy data.
                label = np.zeros((len(ts_sess),), dtype=np.int64)
            else:
                path = Path(
                    cfg.dataset.annotation.path.dir,
                    cfg.dataset.annotation.path.fname
                )
                df_label = optk.data.load_and_resample_operation_labels(
                    path, ts_sess, classes=self.classes)
                label = df_label["act_idx"].values

            data.append({
                "user": user,
                "session": session,
                "data": x_sess,
                "label": label,
                "unixtime": ts_sess,
            })

            seq_len = x_sess.shape[1]
            index += [dict(seq=seq_idx, seg=seg_idx, pos=pos)
                      for seg_idx, pos in enumerate(range(0, seq_len, self.window))]

        self.data = data
        self.index = tuple(index)

    def preprocessing(self):
        """This method is called after ``load_dataset()`` method and apply preprocessing to loaded data.

        Todo:
            - [ ] sklearn.preprocessing.StandardScaler()
            - [ ] DA (half_body_transform)
                - https://github.com/open-mmlab/mmskeleton/blob/b4c076baa9e02e69b5876c49fa7c509866d902c7/mmskeleton/datasets/estimation.py#L62
        """
        logger.warning("No preprocessing is applied.")

    @ property
    def num_classes(self) -> int:
        return len(self.classes)

    def __str__(self) -> str:
        s = (
            "OpenPackKeypoint("
            f"index={len(self.index)}, "
            f"num_sequence={len(self.data)}"
            ")"
        )
        return s

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return self

    def __getitem__(self, index: int) -> Dict:
        seq_idx, seg_idx = self.index[index]["seq"], self.index[index]["seg"]
        seq_dict = self.data[seq_idx]
        seq_len = seq_dict["data"].shape[1]

        # TODO: Make utilities to extract window from long sequence.
        head = seg_idx * self.window
        tail = (seg_idx + 1) * self.window
        if tail >= seq_len:
            pad_tail = tail - seq_len
            tail = seq_len
        else:
            pad_tail = 0
        assert (
            head >= 0) and (
            tail > head) and (
            tail <= seq_len), f"head={head}, tail={tail}"

        x = seq_dict["data"][:, head:tail]
        t = seq_dict["label"][head:tail]
        ts = seq_dict["unixtime"][head:tail]

        if pad_tail > 0:
            x = np.pad(x, [(0, 0), (0, pad_tail), (0, 0)],
                       mode="constant", constant_values=0)
            t = np.pad(t, [(0, pad_tail)], mode="constant",
                       constant_values=self.classes.get_ignore_class_index())
            ts = np.pad(ts, [(0, pad_tail)],
                        mode="constant", constant_values=ts[-1])

        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        ts = torch.from_numpy(ts)
        return {"x": x, "t": t, "ts": ts}


class OpenPackImages(torch.utils.data.Dataset):

    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = pd.read_csv(data)
        self.path = path
        self.transform = transform

    @property
    def num_classes(self) -> int:
        """Returns the number of classes

        Returns:
            int
        """
        return len(self.classes)

    def __str__(self) -> str:
        s = (
            "OpenPackImages()"
        )
        return s

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return self
    def __getitem__(self,index):
        img_name = self.data["id"].iloc[index]
        t = self.data["activity"].iloc[index]
        ts = self.data["unixtime"].iloc[index]

        img_path = os.path.join(self.path, img_name)
        x = img.imread(img_path)
        
        if self.transform is not None:
            x = self.transform(x)

        x = x
        t = t
        ts = ts
        return {"x": x, "t": t, "ts": ts}
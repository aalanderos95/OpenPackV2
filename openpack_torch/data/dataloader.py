"""``dataloader`` provide utility function to load files saved in OpenPack dataset format.
"""
import cv2 as cv
import datetime as dt
import json
import os
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

logger = getLogger(__name__)


def load_keyfpoints(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load keypoints from JSON.

    Args:
        path (Path): path to a target JSON file.
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            * T (np.ndarray): unixtime for each frame.
            * X (np.ndarray): xy-cordinates of keypoints. and the score of corresponding
                prediction. shape=(3, FRAMES, NODE). The first dim is corresponding to
                [x-cordinate, y-cordinate, score].
    Todo:
        * Handle the JSON file that contains keypoints from multiple people.
    """
    with open(path, "r") as f:
        data = json.load(f)
    logger.debug(f"load keypoints from {path}")

    T, X = [], []
    for i, d in enumerate(data["annotations"][:]):
        ut = d.get("image_id", -1)
        kp = np.array(d.get("keypoints", []))

        X.append(kp.T)
        T.append(ut)

    T = np.array(T)
    X = np.stack(X, axis=1)

    return T, X

def loadImage( 
    paths: Union[Tuple[Path, ...], List[Path]],
    names = [],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load keypoints from JSON.

    Args:
        path (Path): path to a target JSON file.
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            * T (np.ndarray): unixtime for each frame.
            * X (np.ndarray): xy-cordinates of keypoints. and the score of corresponding
                prediction. shape=(3, FRAMES, NODE). The first dim is corresponding to
                [x-cordinate, y-cordinate, score].
    Todo:
        * Handle the JSON file that contains keypoints from multiple people.
    """
    X = []
    T = []
    contPaths = 0;
    df = pd.DataFrame(columns=["unixtime"])
    for path in paths:
        img = cv.imread(path)
        X.append(img)
        
        name = names[contPaths].replace(".jpeg","")
        from datetime import datetime
        import time

        d = datetime(int(name[:4]),int(name[4:6]),int(name[6:8]),int(name[9:11]),int(name[11:13]),int(name[13:15]),int(name[16:22]))
        unix = datetime.timestamp(d)
        new_row = pd.DataFrame({'unixtime':np.int64(str(unix).replace(".","")[:13])}, index=[0])
       
        df = pd.concat([df, new_row], ignore_index = True)
        contPaths = contPaths + 1

    T = df["unixtime"].values
    print(T)
    
    return T,X


def load_imu_all(
    paths: Union[Tuple[Path, ...], List[Path]],
    channels=[],
    muestreoN=int,
    hz=[],
    th: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load IMU data from CSVs.

    Args:
        paths (Union[Tuple[Path, ...], List[Path]]): list of paths to target CSV.
            (e.g., [**/atr01/S0100.csv])
        use_acc (bool, optional): include acceleration signal (e.g., ``acc_x, acc_y, acc_z``).
            Defaults to True.
        use_gyro (bool, optional): include gyro scope signal (e.g., ``gyro_x, gyro_y, gyro_z``).
            Defaults to False.
        use_quat (bool, optional): include quaternion data(e.g.,
            ``quat_w, quat_x, quat_y, quat_z``). Defaults to False.
        th (int, optional): threshold of timestamp difference [ms].
            Default. 30 [ms] (<= 1 sample)
    Returns:
        Tuple[np.ndarray, np.ndarray]: unixtime and loaded sensor data.
    """
    assert isinstance(paths, (tuple, list)), (
        f"the first argument `paths` expects tuple of Path, not {type(paths)}."
    )

    ts_ret, x_ret, ts_list = None, [], []
    contPaths = 0
    maxminunixtime = 0
    minmaxunixtime = 0
    muestreo = str(int(1000 / muestreoN)) + 'L'

    for path in paths:
        if (os.path.exists(path)):
            df = pd.read_csv(path)
            logger.debug(f"load IMU data from {path} -> df={df.shape}")

            # NOTE: Error handling : ATR01 in U0101-S0500 has timestamp error.
            #       See an issue #87.
            if str(path).endswith("/U0101/atr/atr01/S0500.csv"):
                df = df.drop(0, axis=0)
                df = df.reset_index(drop=True)

            if "atr" in str(path):
                if hz[contPaths] != (1000 / muestreoN):
                    # RESAMPLE
                    df["Resample"] = df['unixtime'].astype('datetime64[ms]')

                    df = df.set_index('Resample')
                    if ((1000 / muestreoN) < hz[contPaths]):
                        df = df.reset_index().groupby(
                            pd.Grouper(
                                freq=muestreo,
                                key='Resample')).mean(
                            numeric_only=True)
                    else:
                        df = df.reset_index().groupby(
                            pd.Grouper(
                                freq=muestreo,
                                key='Resample')).mean().interpolate(
                            method='linear',
                            limit_direction='both',
                            axis=0)
                    df = df.fillna(method="bfill")
                    df['unixtime'] = df.index.to_series().apply(
                        lambda x: np.int64(str(pd.Timestamp(x).value)[0:13]))
                ts = df["unixtime"].values
            else:
                # Rename Column
                df = df.rename(columns={"time": "unixtime"})
                if hz[contPaths] != (1000 / muestreoN):
                    if (len(df) == 0):
                        # IF IT DOES NOT HAVE DATA, A NEW DATA FRAME OF ZEROS
                        # WILL BE GENERATED WITH THE COLUMN WITH THE CURRENT
                        # UNIX
                        min = str(maxminunixtime)
                        max = str(minmaxunixtime)
                        min = np.int64(min[0:len(min) - 3] + '000')
                        max = np.int64(max[0:len(max) - 3] + '000')
                        arrayUnixTimes = np.arange(min, max, muestreoN)
                        ts_df = pd.DataFrame()
                        ts_df['unixtime'] = pd.Series(arrayUnixTimes)
                        df = pd.concat([df, ts_df], ignore_index=True)
                    # RESAMPLE
                    df.replace(np.nan, 0)
                    df = df.fillna(method="bfill")
                    df["Resample"] = df['unixtime'].astype('datetime64[ms]')
                    df = df.set_index('Resample')
                    if ((1000 / muestreoN) < hz[contPaths]):
                        df = df.reset_index().groupby(
                            pd.Grouper(
                                freq=muestreo,
                                key='Resample')).mean(
                            numeric_only=True)
                    else:
                        df = df.reset_index().groupby(
                            pd.Grouper(
                                freq=muestreo,
                                key='Resample')).mean().interpolate(
                            method='linear',
                            limit_direction='both',
                            axis=0)
                        #df = df.interpolate(method='linear', limit_direction='forward', axis=0)

                    df.replace(np.nan, 0)
                    df = df.fillna(0)
                    df['unixtime'] = df.index.to_series().apply(
                        lambda x: np.int64(str(pd.Timestamp(x).value)[0:13]))
                ts = df["unixtime"].values

            if ts[0] > maxminunixtime:
                maxminunixtime = ts[0]
            if (minmaxunixtime == 0):
                minmaxunixtime = ts[len(ts) - 1]
            if ts[len(ts) - 1] < minmaxunixtime:
                minmaxunixtime = ts[len(ts) - 1]
            x = df[channels[contPaths]].values.T
            ts_list.append(ts)
            x_ret.append(x)
            contPaths = contPaths + 1
        else:
            df = pd.DataFrame(columns=channels[contPaths])
            # COLUMNS
            logger.debug(f"load IMU data from {path} -> df={df.shape}")

            # IF IT DOES NOT HAVE DATA, A NEW DATA FRAME OF ZEROS WILL BE
            # GENERATED WITH THE COLUMN WITH THE CURRENT UNIX
            min = str(maxminunixtime)
            max = str(minmaxunixtime)
            min = np.int64(min[0:len(min) - 3] + '000')
            max = np.int64(max[0:len(max) - 3] + '000')
            arrayUnixTimes = np.arange(min, max, muestreoN)
            ts_df = pd.DataFrame()
            ts_df['unixtime'] = pd.Series(arrayUnixTimes)
            df = pd.concat([df, ts_df], ignore_index=True)
            df.replace(np.nan, 0)
            df = df.fillna(0)
            # RESAMPLE
            df["Resample"] = df['unixtime'].astype('datetime64[ms]')
            df = df.set_index('Resample')
            if ((1000 / muestreoN) < hz[contPaths]):
                df = df.reset_index().groupby(
                    pd.Grouper(
                        freq=muestreo,
                        key='Resample')).mean(
                    numeric_only=True)
            else:
                df = df.reset_index().groupby(
                    pd.Grouper(
                        freq=muestreo,
                        key='Resample')).mean().interpolate(
                    method='linear',
                    limit_direction='forward',
                    axis=0)
                #df = df.interpolate(method='linear', limit_direction='forward', axis=0)

            df = df.replace(np.nan, 0)
            df = df.fillna(0)
            df['unixtime'] = df.index.to_series().apply(
                lambda x: np.int64(str(pd.Timestamp(x).value)[0:13]))
            ts = df["unixtime"].values
            if ts[0] > maxminunixtime:
                maxminunixtime = ts[0]
            if (minmaxunixtime == 0):
                minmaxunixtime = ts[len(ts) - 1]
            if ts[len(ts) - 1] < minmaxunixtime:
                minmaxunixtime = ts[len(ts) - 1]

            x = df[channels[contPaths]].values.T
            ts_list.append(ts)
            x_ret.append(x)
            contPaths = contPaths + 1
    ts_ret = None

    fT = False
    newts_list = []
    newx_ret = []
    ts_list_reshape = []
    for i in range(len(x_ret)):
        x_ret_reshape = []
        for j in range(len(x_ret[i])):
            X_train = pd.DataFrame()
            X_trainMean = pd.DataFrame()
            ts_train = pd.DataFrame()
            ts_trainMean = pd.DataFrame()
            data = {'unixtime': ts_list[i], 'value': x_ret[i][j, :]}
            df = pd.DataFrame(data=data)
            dfMean = df.query(
                f"`unixtime` >= {maxminunixtime} and `unixtime` <={minmaxunixtime}")

            x_list = list(x for x in dfMean['value'].to_numpy())
            tsN_list = list(x for x in dfMean['unixtime'].to_numpy())

            X_train['value'] = pd.Series(x_list)
            ts_train['unixtime'] = pd.Series(tsN_list)
            X_trainMean = pd.concat([X_trainMean, X_train], ignore_index=True)
            ts_trainMean = pd.concat(
                [ts_trainMean, ts_train], ignore_index=True)

            if not fT:
                #x_ret_reshape = [len(x_ret),len(X_trainMean)]
                ts_list_reshape = ts_trainMean.values.T
                fT = True
            x_ret_reshape.append(X_trainMean.values.T[0])
        newts_list.append(ts_list_reshape[0])
        newx_ret.append(x_ret_reshape)
    x_ret = newx_ret
    ts_list = newts_list

    for i in range(contPaths):
        #x_ret[i] = x_ret[i][:, :min_len]
        #ts_list[i] = ts_list[i][:min_len]

        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], "
                f"but difference smaller than th={th} is allowed."
            )
        """logger.info(f"Start resample for path: {paths[i]}.")
        x_ret[i] = resample(arrayUnixTimes,x_ret[i], ts_list[i])
        logger.info(f"Finish resample for path: {paths[i]}.")
        ts_list[i] = arrayUnixTimes

        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], "
                f"but difference smaller than th={th} is allowed."
            )
        """
    x_ret = np.concatenate(x_ret, axis=0)
    return ts_ret, x_ret


def load_imu_new(
    paths: Union[Tuple[Path, ...], List[Path]],
    pathsWOSession=[],
    channels=[],
    muestreoN=int,
    hz=[],
    statistics: bool = False,
    th: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load IMU data from CSVs.

    Args:
        paths (Union[Tuple[Path, ...], List[Path]]): list of paths to target CSV.
            (e.g., [**/atr01/S0100.csv])
        use_acc (bool, optional): include acceleration signal (e.g., ``acc_x, acc_y, acc_z``).
            Defaults to True.
        use_gyro (bool, optional): include gyro scope signal (e.g., ``gyro_x, gyro_y, gyro_z``).
            Defaults to False.
        use_quat (bool, optional): include quaternion data(e.g.,
            ``quat_w, quat_x, quat_y, quat_z``). Defaults to False.
        th (int, optional): threshold of timestamp difference [ms].
            Default. 30 [ms] (<= 1 sample)
    Returns:
        Tuple[np.ndarray, np.ndarray]: unixtime and loaded sensor data.
    """
    assert isinstance(paths, (tuple, list)), (
        f"the first argument `paths` expects tuple of Path, not {type(paths)}."
    )

    ts_ret, x_ret, ts_list = None, [], []
    contPaths = 0

    sessions = [
        'S0100.csv',
        'S0200.csv',
        'S0300.csv',
        'S0400.csv',
        'S0500.csv']

    series = ['mean', 'std', 'max','min','median']
    for path in paths:
        if (os.path.exists(path)):
            df = pd.read_csv(path)
            logger.debug(f"load IMU data from {path} -> df={df.shape}")
            assert set(channels[contPaths]) < set(df.columns)

            # NOTE: Error handling : ATR01 in U0101-S0500 has timestamp error.
            #       See an issue #87.
            if str(path).endswith("/U0101/atr/atr01/S0500.csv"):
                df = df.drop(0, axis=0)
                df = df.reset_index(drop=True)

            df = df.rename(columns={"time": "unixtime"})
        
            ts = df["unixtime"].values
            x = df[channels[contPaths]].values.T

            ts_list.append(ts)
            x_ret.append(x)
            contPaths = contPaths + 1
        else:
            df = pd.DataFrame(columns=channels[contPaths])
            ts_df = pd.DataFrame(columns=["unixtime"])
            df = pd.concat([df, ts_df], ignore_index=True)
            x = df[channels[contPaths]].values.T
            ts = df["unixtime"].values
            ts_list.append(ts)
            x_ret.append(x)
            contPaths = contPaths + 1

    #RESAMPLE
    muestreo = str(int(1000 / muestreoN)) + 'L'
    for i in range(len(paths)):
        if i != 0:
            df = pd.DataFrame()
            df['unixtime'] = pd.Series(ts_list[i])
            for channel in range(len(channels[i])):
                df[channels[i][channel]] = pd.Series(x_ret[i][channel, :])

            if (len(df) == 0):
                ts_df = pd.DataFrame()
                ts_df['unixtime'] = pd.Series(ts_list[0])
                df = pd.concat([df, ts_df])
                for session in sessions:
                    pathNew = Path(
                        pathsWOSession[i],
                        session
                    )
                    if (os.path.exists(pathNew)):
                        dfNew = pd.read_csv(pathNew)
                        if (len(dfNew) != 0):
                            for channel in channels[i]:
                                mean_value = dfNew[channel].mean()
                                df[channel].fillna(
                                    value=0, inplace=True)
                            break
                
                if(statistics):
                    if len(channels[i]) > 1:
                        #statistics            
                        df['mean'] = df[channels[i]].mean(axis=1, numeric_only=True);
                        df['std'] = df[channels[i]].std(axis=1, numeric_only=True)
                        df['max'] = df[channels[i]].max(axis=1, numeric_only=True)
                        df['min'] = df[channels[i]].min(axis=1, numeric_only=True)
                        df['median'] = df[channels[i]].median(axis=1, numeric_only=True)

                        channels[i] += series;
                        df = df.fillna(0)

                ts = df["unixtime"].values
                x = df[channels[i]].values.T
                ts_list[i] = ts
                x_ret[i] = x
            else:
                
                ts_df = pd.DataFrame()
                ts_df['unixtime'] = pd.Series(ts_list[0])

                df["Resample"] = df['unixtime'].astype('datetime64[ms]')
                df = df.set_index('Resample')

                df = df.reset_index().groupby(
                    pd.Grouper(
                        freq=muestreo,
                        key='Resample')).mean().interpolate(
                    method='linear',
                    limit_direction='both',
                    axis=0)
                for channel in channels[i]:
                    mean_value = df[channel].mean()
                    df[channel].fillna(value=0, inplace=True)

                df['unixtime'] = df.index.to_series().apply(
                    lambda x: np.int64(str(pd.Timestamp(x).value)[0:13]))
                df_vacio = df
                df = concatDf(
                    df,
                    ts_df,
                    df_vacio,
                    muestreo,
                    hz[i],
                    muestreoN,
                    channels[i])
                
                if(statistics):
                    if len(channels[i]) > 1:
                        #statistics            
                        df['mean'] = df[channels[i]].mean(axis=1, numeric_only=True);
                        df['std'] = df[channels[i]].std(axis=1, numeric_only=True)
                        df['max'] = df[channels[i]].max(axis=1, numeric_only=True)
                        df['min'] = df[channels[i]].min(axis=1, numeric_only=True)
                        df['median'] = df[channels[i]].median(axis=1, numeric_only=True)

                        channels[i] += series;
                        df = df.fillna(0)

                ts = df["unixtime"].values
                x = df[channels[i]].values.T
                ts_list[i] = ts
                x_ret[i] = x
        else:
            df = pd.DataFrame()
            df['unixtime'] = pd.Series(ts_list[i])
            for channel in range(len(channels[i])):
                df[channels[i][channel]] = pd.Series(x_ret[i][channel, :])

            df["Resample"] = df['unixtime'].astype('datetime64[ms]')
            df = df.set_index('Resample')
            df = df.reset_index().groupby(
                pd.Grouper(
                    freq=muestreo,
                    key='Resample')).mean().interpolate(
                method='linear',
                limit_direction='both',
                axis=0)    
            
            
            for channel in channels[i]:
                mean_value = df[channel].mean()
                df[channel].fillna(value=0, inplace=True)

            df['unixtime'] = df.index.to_series().apply(
                lambda x: np.int64(str(pd.Timestamp(x).value)[0:13]))

            ts = df["unixtime"].values
            
            if(statistics):
                if len(channels[i]) > 1:
                    #statistics            
                    df['mean'] = df[channels[i]].mean(axis=1, numeric_only=True)
                    df['std'] = df[channels[i]].std(axis=1, numeric_only=True)
                    df['max'] = df[channels[i]].max(axis=1, numeric_only=True)
                    df['min'] = df[channels[i]].min(axis=1, numeric_only=True)
                    df['median'] = df[channels[i]].median(axis=1, numeric_only=True)

                    channels[i] += series;
                    df = df.fillna(0)


            x = df[channels[i]].values.T
            ts_list[i] = ts
            x_ret[i] = x

    ts_ret = None
    for i in range(len(paths)):
        #x_ret[i] = x_ret[i][:, :min_len]
        #ts_list[i] = ts_list[i][:min_len]

        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], "
                f"but difference smaller than th={th} is allowed."
                f"num path is {paths[i]}."
            )
    
    x_ret = np.concatenate(x_ret, axis=0)
    return ts_ret, x_ret


def concatDf(
    df: pd.DataFrame,
    ts_n: pd.DataFrame,
    df_vacio: pd.DataFrame,
    muestreo: str,
    hzAct: int,
    muestreoN: int,
    channels=[]
) -> pd.DataFrame:
    df_vacio = df_vacio.iloc[0:0]
    
    df_vacio = pd.concat([df_vacio, ts_n])
   
    #df_vacio = df.loc[df.unixtime.isin(df_n.unixtime)]
    #df_vacio = df.merge(df_vacio, how='inner', left_index=True, right_index=True)
    """import time
    inicio = time.time()

    for i in range(len(df_n)):
        unixActual = str(df_n["unixtime"].iloc[i])
        dfFind = df.query(f"`unixtime` == {unixActual}")

        if len(dfFind != 0):
            df_vacio.loc[i] = dfFind.iloc[0]

    fin = time.time()

    logger.info(f"Tiempo de Ejecución: {(fin-inicio)}!")


    for channel in channels:
        mean_value=df_vacio[channel].mean()
        df_vacio[channel].fillna(value=mean_value, inplace=True)   """
    df_vacio = ts_n.merge(df, on="unixtime", how="outer")
   
    unixInicial = str(ts_n["unixtime"].iloc[0])
    unixFinal = str(ts_n["unixtime"].iloc[len(ts_n) - 1])
    df_vacio = df_vacio.query(
        f"`unixtime` >= {unixInicial} and `unixtime` <={unixFinal}")
    df_vacio["Resample"] = df_vacio['unixtime'].astype('datetime64[ms]')
    # print(df_vacioPrueba)
    #df_vacio['Resample'] = pd.to_datetime(df_vacio.unixtime, unit='ms')
    # print(df_vacio)
    # exit();
    df_vacio = df_vacio.set_index('Resample')

    df_vacio = df_vacio.reset_index().groupby(
        pd.Grouper(
            freq=muestreo,
            key='Resample')).mean().interpolate(
        method='linear',
        limit_direction='both',
        axis=0)

    df_vacio['unixtime'] = df_vacio.index.to_series().apply(
        lambda x: np.int64(str(pd.Timestamp(x).value)[0:13]))

    for channel in channels:
        mean_value = df_vacio[channel].mean()
        df_vacio[channel].fillna(value=mean_value, inplace=True)

    return df_vacio


def remuestrear(
    xs: np.ndarray,
    unixtimes: np.ndarray,
    maxminunixtime_ms: np.int64,
    minmaxunixtime_ms: np.int64,
) -> Tuple[np.ndarray, np.ndarray, np.int64]:
    # Inicializar maxminunixtime_ms a 0
    restInt = 0
    # Remuestrear unixtimes
    for i in range(len(unixtimes)):
        if (unixtimes[i] > maxminunixtime_ms):
            unixtimes = unixtimes[i:]
            xs = xs[:, i:]
            restInt = i
            break

    for i in range(len(unixtimes), 0, -1):
        if (unixtimes[i - 1] < minmaxunixtime_ms):
            unixtimes = unixtimes[:i - 1]
            xs = xs[:, :i - 1]
            restInt = restInt + (i - 1)
            break

    return xs, unixtimes, restInt


def resample(
    unixtimesFinal: np.ndarray,
    xs: np.ndarray,
    unixTimesActual: np.ndarray,
) -> Tuple[np.ndarray]:
    xs_return = np.empty([len(xs), len(unixtimesFinal) - 1])

    for x in range(len(xs)):
        X_train = pd.DataFrame()
        X_trainMean = pd.DataFrame()
        data = {'unixtime': unixTimesActual, 'value': xs[x, :]}
        df = pd.DataFrame(data=data, index=unixTimesActual)

        for unix in range(len(unixtimesFinal) - 1):
            x_list = []
            dfMean = df.query(
                f"`unixtime` >= {unixtimesFinal[unix]} and `unixtime` <={unixtimesFinal[unix+1]}")
            x_list.append(dfMean.value)
            # Obtener Medias
            X_train['x_mean'] = pd.Series(x_list).apply(
                lambda value: value.mean())
            # X_trainMean.concat(X_train, ignore_index = True)
            X_trainMean = pd.concat([X_trainMean, X_train], ignore_index=True)
        xs_return[x] = X_trainMean.transpose()

    return xs_return


def load_imu(
    paths: Union[Tuple[Path, ...], List[Path]],
    channels=[],
    th: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load IMU data from CSVs.

    Args:
        paths (Union[Tuple[Path, ...], List[Path]]): list of paths to target CSV.
            (e.g., [**/atr01/S0100.csv])
        use_acc (bool, optional): include acceleration signal (e.g., ``acc_x, acc_y, acc_z``).
            Defaults to True.
        use_gyro (bool, optional): include gyro scope signal (e.g., ``gyro_x, gyro_y, gyro_z``).
            Defaults to False.
        use_quat (bool, optional): include quaternion data(e.g.,
            ``quat_w, quat_x, quat_y, quat_z``). Defaults to False.
        th (int, optional): threshold of timestamp difference [ms].
            Default. 30 [ms] (<= 1 sample)
    Returns:
        Tuple[np.ndarray, np.ndarray]: unixtime and loaded sensor data.
    """
    assert isinstance(paths, (tuple, list)), (
        f"the first argument `paths` expects tuple of Path, not {type(paths)}."
    )

    cont = 0
    ts_ret, x_ret, ts_list = None, [], []
    for path in paths:
        df = pd.read_csv(path)
        logger.debug(f"load IMU data from {path} -> df={df.shape}")
        assert set(channels[cont]) < set(df.columns)

        # NOTE: Error handling : ATR01 in U0101-S0500 has timestamp error.
        #       See an issue #87.
        if str(path).endswith("/U0101/atr/atr01/S0500.csv"):
            df = df.drop(0, axis=0)
            df = df.reset_index(drop=True)

        df = df.rename(columns={"time": "unixtime"})
        ts = df["unixtime"].values
        x = df[channels[cont]].values.T

        ts_list.append(ts)
        x_ret.append(x)
        cont = cont + 1

    min_len = min([len(ts) for ts in ts_list])
    ts_ret = None
    for i in range(len(paths)):
        x_ret[i] = x_ret[i][:, :min_len]
        ts_list[i] = ts_list[i][:min_len]

        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], "
                f"but difference smaller than th={th} is allowed."
            )

    x_ret = np.concatenate(x_ret, axis=0)
    return ts_ret, x_ret

def load_imu_ant(
    paths: Union[Tuple[Path, ...], List[Path]],
    use_acc: bool = True,
    use_gyro: bool = False,
    use_quat: bool = False,
    th: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load IMU data from CSVs.
    Args:
        paths (Union[Tuple[Path, ...], List[Path]]): list of paths to target CSV.
            (e.g., [**/atr01/S0100.csv])
        use_acc (bool, optional): include acceleration signal (e.g., ``acc_x, acc_y, acc_z``).
            Defaults to True.
        use_gyro (bool, optional): include gyro scope signal (e.g., ``gyro_x, gyro_y, gyro_z``).
            Defaults to False.
        use_quat (bool, optional): include quaternion data(e.g.,
            ``quat_w, quat_x, quat_y, quat_z``). Defaults to False.
        th (int, optional): threshold of timestamp difference [ms].
            Default. 30 [ms] (<= 1 sample)
    Returns:
        Tuple[np.ndarray, np.ndarray]: unixtime and loaded sensor data.
    """
    assert isinstance(paths, (tuple, list)), (
        f"the first argument `paths` expects tuple of Path, not {type(paths)}."
    )

    channels = []
    if use_acc:
        channels += ["acc_x", "acc_y", "acc_z"]
    if use_gyro:
        channels += ["gyro_x", "gyro_y", "gyro_z"]
    if use_quat:
        channels += ["quat_w", "quat_x", "quat_y", "quat_z"]

    ts_ret, x_ret, ts_list = None, [], []
    for path in paths:
        df = pd.read_csv(path)
        logger.debug(f"load IMU data from {path} -> df={df.shape}")
        assert set(channels) < set(df.columns)

        # NOTE: Error handling : ATR01 in U0101-S0500 has timestamp error.
        #       See an issue #87.
        if str(path).endswith("/U0101/atr/atr01/S0500.csv"):
            df = df.drop(0, axis=0)
            df = df.reset_index(drop=True)

        ts = df["unixtime"].values
        x = df[channels].values.T

        ts_list.append(ts)
        x_ret.append(x)

    min_len = min([len(ts) for ts in ts_list])
    ts_ret = None
    for i in range(len(paths)):
        x_ret[i] = x_ret[i][:, :min_len]
        ts_list[i] = ts_list[i][:min_len]

        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], "
                f"but difference smaller than th={th} is allowed."
            )

    x_ret = np.concatenate(x_ret, axis=0)
    return ts_ret, 

def load_e4acc(
    paths: Union[Tuple[Path, ...], List[Path]],
    th: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load IMU data from CSVs.
    Args:
        paths (Union[Tuple[Path, ...], List[Path]]): list of paths to target CSV.
            (e.g., [**/atr01/S0100.csv])
        use_acc (bool, optional): include acceleration signal (e.g., ``acc_x, acc_y, acc_z``).
            Defaults to True.
        use_gyro (bool, optional): include gyro scope signal (e.g., ``gyro_x, gyro_y, gyro_z``).
            Defaults to False.
        use_quat (bool, optional): include quaternion data(e.g.,
            ``quat_w, quat_x, quat_y, quat_z``). Defaults to False.
        th (int, optional): threshold of timestamp difference [ms].
            Default. 30 [ms] (<= 1 sample)
    Returns:
        Tuple[np.ndarray, np.ndarray]: unixtime and loaded sensor data.
    """
    assert isinstance(paths, (tuple, list)), (
        f"the first argument `paths` expects tuple of Path, not {type(paths)}."
    )

    channels = ["acc_x", "acc_y", "acc_z"]
    
    ts_ret, x_ret, ts_list = None, [], []
    for path in paths:
        if (os.path.exists(path)):    
            df = pd.read_csv(path)
            if df.empty:
                return None, None
            logger.debug(f"load IMU data from {path} -> df={df.shape}")
            assert set(channels) < set(df.columns)

            # NOTE: Error handling : ATR01 in U0101-S0500 has timestamp error.
            #       See an issue #87.
            if str(path).endswith("/U0101/atr/atr01/S0500.csv"):
                df = df.drop(0, axis=0)
                df = df.reset_index(drop=True)

            ts = df["time"].values
            x = df[channels].values.T

            ts_list.append(ts)
            x_ret.append(x)
        else:
            return None, None
    
    min_len = min([len(ts) for ts in ts_list])
    ts_ret = None
    for i in range(len(paths)):
        x_ret[i] = x_ret[i][:, :min_len]
        ts_list[i] = ts_list[i][:min_len]

        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            if delta.max() > th:
                return None, None
            """assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], "
                f"but difference smaller than th={th} is allowed."
            )"""
    x_ret = np.concatenate(x_ret, axis=0)

    return x_ret, ts_ret 

def load_e4acc_alineado(
    paths: Union[Tuple[Path, ...], List[Path]],
    th: int = 33,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load IMU data from CSVs.
    Args:
        paths (Union[Tuple[Path, ...], List[Path]]): list of paths to target CSV.
            (e.g., [**/atr01/S0100.csv])
        use_acc (bool, optional): include acceleration signal (e.g., ``acc_x, acc_y, acc_z``).
            Defaults to True.
        use_gyro (bool, optional): include gyro scope signal (e.g., ``gyro_x, gyro_y, gyro_z``).
            Defaults to False.
        use_quat (bool, optional): include quaternion data(e.g.,
            ``quat_w, quat_x, quat_y, quat_z``). Defaults to False.
        th (int, optional): threshold of timestamp difference [ms].
            Default. 30 [ms] (<= 1 sample)
    Returns:
        Tuple[np.ndarray, np.ndarray]: unixtime and loaded sensor data.
    """
    assert isinstance(paths, (tuple, list)), (
        f"the first argument `paths` expects tuple of Path, not {type(paths)}."
    )

    channels = ["acc_x", "acc_y", "acc_z"]
    
    ts_ret, x_ret, ts_list = None, [], []
    for path in paths:
        if (os.path.exists(path)):            
            df = pd.read_csv(path)
            if df.empty:
                return None, None
            logger.debug(f"load IMU data from {path} -> df={df.shape}")
            assert set(channels) < set(df.columns)

            # NOTE: Error handling : ATR01 in U0101-S0500 has timestamp error.
            #       See an issue #87.
            if str(path).endswith("/U0101/atr/atr01/S0500.csv"):
                df = df.drop(0, axis=0)
                df = df.reset_index(drop=True)

            ts = df["time"].values
            x = df[channels].values.T

            ts_list.append(ts)
            x_ret.append(x)
        else:
            return None, None

    x_ret, ts_list = alinear(x_ret, ts_list)
    min_len = min([len(ts) for ts in ts_list])
    ts_ret = None
    
    for i in range(len(paths)):
        x_ret[i] = x_ret[i][:, :min_len]        
        ts_list[i] = ts_list[i][:min_len]
        
        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            
            assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], "
                f"but difference smaller than th={th} is allowed."
            )
    x_ret = np.concatenate(x_ret, axis=0)
    return x_ret, ts_ret     

def alinear(x_list, ts_list):
    mints = None
    maxts = None
    for ts in ts_list:
        if mints == None:
            mints = min(ts)
            maxts = max(ts)
        else:
            if min(ts) > mints:
                mints = min(ts)
            if max(ts) < maxts:
                maxts = max(ts)
    
    for i in range(len(ts_list)):
        use_index = np.logical_and(ts_list[i] >= mints, ts_list[i] <= maxts)
        ts_list[i] = ts_list[i][use_index]
        x_list[i] = x_list[i][:, use_index]   

    return x_list, ts_list
def load_and_resample_scan_log(
    path: Path,
    unixtimes_ms: np.ndarray,
) -> np.ndarray:
    """Load scan log data such as HT, and make binary vector for given timestamps.
    Elements that have the same timestamp in second precision are marked as 1.
    Other values are set to 0.

    Args:
        path (Path): path to a scan log CSV file.
        unixtimes_ms (np.ndarray):  unixtime seqeuence (milli-scond precision).
            shape=(T,).

    Returns:
        np.ndarray: binary 1d vector.
    """
    assert unixtimes_ms.ndim == 1
    df = pd.read_csv(path)
    logger.info(f"load scan log from {path} -> df={df.shape}")

    unixtimes_sec = unixtimes_ms // 1000

    X_log = np.zeros(len(unixtimes_ms)).astype(np.int32)
    for utime_ms in df["unixtime"].values:
        utime_sec = utime_ms // 1000
        ind = np.where(unixtimes_sec == utime_sec)[0]
        X_log[ind] = 1

    return X_log

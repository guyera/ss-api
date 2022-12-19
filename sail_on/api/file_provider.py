"""A mock provider class for testing the Server class and Provider interface."""

from .provider import Provider, FileResult
from .errors import ServerError, ProtocolError, RoundError
from .constants import ProtocolConstants

import pandas as pd

import logging
import os
import glob
import csv
import uuid
import datetime
import json
import traceback
import nltk
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from dateutil import parser
import zipfile
import re

from typing import List, Optional, Dict, Any
from csv import reader
from io import BytesIO
from cachetools import LRUCache, cached

@cached(cache=LRUCache(maxsize=32))
def read_gt_csv_file(file_location, with_header=False):
    idx = 0 if with_header else 1
    with open(file_location, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar='|')
        return [x for x in csv_reader][idx:]

@cached(cache=LRUCache(maxsize=128))
def read_meta_data(file_location):
    with open(file_location, "r") as md:
        return json.load(md)

# region Session log related functions
def get_session_info(folder: str, session_id: str, in_process_only: bool = True) -> Dict[str, Any]:
    """Retrieve session info."""
    path = os.path.join(folder, f"{str(session_id)}.json")
    if os.path.exists(path):
        with open(path, "r") as session_file:
            info = json.load(session_file)
            terminated =  "termination" in info
            if terminated and in_process_only:
                raise ProtocolError(
                    "SessionEnded", 
                    "The session being requested has already been terminated. Please either create a new session or request a different ID",
                )
            else:
                return info
    return {}

def get_session_test_info(folder: str, session_id: str, test_id: str) -> Dict[str, Any]:
    path = os.path.join(folder, f"{str(session_id)}.{test_id}.json")
    if os.path.exists(path):
        with open(path, "r") as session_file:
            info = json.load(session_file)
            if "completion" in info:
                raise ProtocolError(
                    "TestCompleted", 
                    "The test being requested has already been completed for this session",
                )
            else:
                return info
    return {}

def write_session_log_file(structure: Dict, filepath: str) -> None:
    with open(filepath, "w") as session_file:
            json.dump(structure, session_file, indent=2)

def log_session(
    folder: str,
    session_id: str,
    activity: str,
    test_id: Optional[str] = None,
    round_id: Optional[int] = None,
    content: Optional[Dict[str, Any]] = None,
    content_loc: Optional[str] = "round",
    return_structure: Optional[bool] = False
) -> Optional[Dict]:
    """Create a log files of all session activity."""
    structure = get_session_info(folder, session_id)
    write_session_file = True
    if test_id is None:
        structure[activity] = {"time": [str(datetime.datetime.now())]}
        if content is not None:
            structure[activity].update(content)
    else:
        test_structure = get_session_test_info(folder, session_id, test_id)
        if activity not in test_structure:
            test_structure[activity] = {"time": [str(datetime.datetime.now())]}
        if content_loc == "activity":
            if content is not None:
                test_structure[activity].update(content)
        if round_id is not None:
            round_id = str(round_id)
            rounds = test_structure[activity].get("rounds", {})
            if round_id not in rounds:
                rounds[round_id] = {"time": [str(datetime.datetime.now())]}
            else:
                rounds[round_id]["time"].append(str(datetime.datetime.now()))
            if content_loc == "round":
                if content is not None:
                    rounds[round_id].update(content)
            test_structure[activity]["rounds"] = rounds
            test_structure[activity]["last round"] = round_id

        if not return_structure:
            write_session_log_file(test_structure, os.path.join(folder, f"{str(session_id)}.{str(test_id)}.json"))
        
        if activity == "completion":    
            session_tests = structure.get("tests", {"completed_tests": []})
            session_tests["completed_tests"].append(test_id)
            structure["tests"] = session_tests
        else:
            write_session_file = False

    if write_session_file:
        write_session_log_file(structure, os.path.join(folder, f"{str(session_id)}.json"))

    if return_structure:
        return test_structure

# endregion

# region Feedback related functions
def read_feedback_file(
        csv_reader: reader,
        feedback_ids: List[str],
        metadata: Dict[str, Any],
        check_constrained=True
) -> Dict[str, str]:
    """
        Gets the feedback data out of the provided
        csv feedback file for the specified ids in 
        the last submitted round.
    """
    feedback_constrained = metadata.get('feedback_constrained', True)

    lines = [x for x in csv_reader]

    try:
        if (not check_constrained or not feedback_constrained) :
            start = 0
            end = len(lines)
        else:
            # under the constrained case, we always look at the last round
            start = len(lines) - int(metadata["round_size"])
            end = start + int(metadata["round_size"])
    except KeyError:
        raise RoundError(
            "no_defined_rounds",
            "round_size not defined in metadata.",
            traceback.format_stack(),
        )

    if feedback_ids is not None and len(feedback_ids) != 0:
        return {
            x[0]: [n for n in x[1:]]
            for x in [[n.strip(" \"'") for n in y] for y in lines][start:end]
            if x[0] in feedback_ids
        }
    else:
        return {
            x[0]: [n for n in x[1:]]
            for x in
            [[n.strip(" \"'") for n in y] for y in lines][start:end]
        }

def get_classification_feedback(
        gt_file: str,
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for classification type feedback"""
    if (feedback_ids is None or len(feedback_ids) == 0):
        # if feedback ids not provided, limit to those in the last round
        with open(result_files[0], "r") as rf:
            result_reader = csv.reader(rf, delimiter=",")
            results = read_feedback_file(result_reader, None, metadata, check_constrained=True)
            feedback_max_ids = min(metadata.get('feedback_max_ids',len(results)),len(results))
            feedback_ids = list(results.keys())[:int(feedback_max_ids)]

    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata,
                                      check_constrained= feedback_ids is None or len(feedback_ids) == 0)

    return {
        x: min(int(ground_truth[x][metadata["columns"][0]]), metadata["known_classes"]) 
        for x in ground_truth.keys()
    }

def get_classification_feedback_topk(
        gt_file: str,
        topk_file: str,
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for classification type feedback"""
    if (feedback_ids is None or len(feedback_ids) == 0):
        # if feedback ids not provided, limit to those in the last round
        with open(result_files[0], "r") as rf:
            result_reader = csv.reader(rf, delimiter=",")
            results = read_feedback_file(result_reader, None, metadata, check_constrained=True)
            feedback_max_ids = min(metadata.get('feedback_max_ids',len(results)),len(results))
            feedback_ids = list(results.keys())[:int(feedback_max_ids)]

    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata,
                                      check_constrained= feedback_ids is None or len(feedback_ids) == 0)
    
    topk = read_feedback_file(read_gt_csv_file(topk_file, with_header=True), None, metadata,
                                      check_constrained= feedback_ids is None or len(feedback_ids) == 0)   
    
    '''
    
    '''
    data = {
        x: topk[str(int(float(ground_truth[x][metadata["columns"][0] - 1])))] 
        for x in ground_truth.keys()
    }
    return data


def get_classification_var_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Grabs and returns"""
    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata,
                                      check_constrained= feedback_ids is None or len(feedback_ids) == 0)

    return {
        x: ground_truth[x][metadata["columns"][0]:metadata["columns"][1]]
        for x in ground_truth.keys()
    }

def get_detection_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any],
    is_given_detection_mode: bool,
) -> Dict[str, Any]:
    """Grabs and returns"""
    if (feedback_ids is None or len(feedback_ids) == 0):
        # if feedback ids not provided, limit to those in the last round
        with open(result_files[0], "r") as rf:
            result_reader = csv.reader(rf, delimiter=",")
            results = read_feedback_file(result_reader, None, metadata, check_constrained=True)
            # if empty feedback ids requested for system detection use budget
            if not is_given_detection_mode:
                feedback_max_ids = min(metadata.get('max_detection_feedback_ids',len(results)),len(results))
                feedback_ids = list(results.keys())[:int(feedback_max_ids)]
            else:
                # No budget for given detection
                feedback_ids = list(results.keys())

    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata,
                                      check_constrained= feedback_ids is None or len(feedback_ids) == 0)

    return {
        x: ground_truth[x][metadata["columns"][0]] for x in ground_truth.keys()
    }

def get_classificaton_score_feedback(
        gt_file: str,
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates and returns feedback on the accuracy of classification"""

    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), None, metadata, check_constrained=False)
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, None, metadata, check_constrained=False)

    # Go through results and count number correct
    num_correct = 0
    for id in results.keys():
        r = int(np.argmax([float(i) for i in results[id]], axis=0))
        g = int(ground_truth[id][metadata["columns"][0]])
        if r == g:
            num_correct += 1
    
    accuracy = float(num_correct) / float(len(results.keys()))
    return {'accuracy' : accuracy}

def get_characterization_feedback(
        gt_file: str,
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for characterization type feedback"""
    # Not implemented
    raise NameError('Characterization Feedback is not supported.')
    known_classes = int(metadata["known_classes"]) + 1

    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata)
    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata, check_constrained=False)


    # If ground truth is not novel, returns 1 is prediction is correct, 
    # otherwise returns 1 if prediction is not a known class
    return {
        x: 0
        if (sum(ground_truth[x][1:known_classes]) > (ground_truth[x][0] + sum(ground_truth[x][known_classes:]))
            and ground_truth[x].index(max(ground_truth[x])) != results[x].index(max(results[x])))
        or (sum(ground_truth[x][1:known_classes]) < (ground_truth[x][0] + sum(ground_truth[x][known_classes:]))
            and (results[x].index(max(results[x])) in range(1, known_classes)))
        else 1
        for x in ground_truth.keys()
    }

def ensure_space(input_str): 
    return ' '.join([x.strip() for x in re.split(r'(\W+)',input_str.replace(';','').replace('"','').replace('|','').replace('  ',' '))])

def get_levenshtein_feedback(
        gt_file: str,
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for levenshtein type feedback"""
    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata, check_constrained=False)
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata)

    return {
        x: [
            nltk.edit_distance(
                ensure_space(ground_truth[x][metadata["columns"][i]]), 
                ensure_space(results[x][0])
            ) 
            for i,_ in enumerate(metadata["columns"])
        ]
        for x in results.keys()
    }

def get_cluster_feedback(
        gt_file: str,
        result_files: List[str],
        feedback_ids: List[str],
        metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates and returns the proper feedback for cluster type feedback"""
    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata, check_constrained=False)
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata)

    if feedback_ids is None:
        feedback_ids = ground_truth.keys()

    # clear ground truth of all but relevant columns
    for key in ground_truth.keys():
        ground_truth[key] = [ground_truth[key][i] for i in metadata["columns"]]

    gt_list = []
    r_list = []
    try:
        for key in sorted(feedback_ids):
            gt_list.append([float(x) for x in ground_truth[key]])
            r_list.append([float(x) for x in results[key]])
    except:
        raise ServerError("MissingIds", "Some requested Ids are missing from either ground truth or results file for the current round")
    
    gt_np = np.array(gt_list).reshape(len(gt_list))
    r_np = np.argmax(np.array(r_list), axis=1)

    return_dict = {
        "nmi": normalized_mutual_info_score(gt_np, r_np)
    }

    for i in np.unique(r_np):
        places = np.where(r_np == i)[0]
        return_dict[str(i)] = (max(np.unique(gt_np[places],return_counts=True)[1])/places.shape[0])
    
    return return_dict

def psuedo_label_feedback(
        gt_file: str,
        feedback_ids: List[str],
        feedback_type: str,
        metadata: Dict[str, Any],
        folder: str,
        session_id: str
) -> Dict[str, Any]:
    "Grabs psuedo label feedback for requested ids"
    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata)

    structure = get_session_info(folder, session_id)

    if "psuedo_labels" in structure:
        if feedback_type in structure["psuedo_labels"]:
            labels = structure["psuedo_labels"][feedback_type]
        else:
            structure["psuedo_labels"][feedback_type] = []
            labels = []
    else:
        structure["psuedo_labels"] = {feedback_type: []}
        labels = []

    return_dict = {}
    for x in ground_truth.keys():
        col = int(ground_truth[x][metadata["columns"][0]])
        if col not in labels:
            labels.append(col)
        return_dict[x] = labels.index(col)

    structure["psuedo_labels"][feedback_type] = labels
    write_session_log_file(structure, os.path.join(folder, f"{str(session_id)}.json"))

    return return_dict

# endregion

def get_hint_typeA(
        gt_file: str,
        metadata: Dict[str, Any],
        hint_file
) -> Dict[str, Any]:
    df = pd.read_csv(hint_file)
    hint = df[df['test_id'] == metadata['curr_test_id']]['novelty_kind'][0]
    return {metadata['curr_test_id'] : hint}

def get_hint_typeB(
        gt_file: str,
        metadata: Dict[str, Any],
        hint_file
) -> Dict[str, Any]:
    df = pd.read_csv(gt_file)
    try:
            round_pos = int(metadata["curr_round_id"]) * int(metadata["round_size"])
    except KeyError:
            raise RoundError(
                "no_defined_rounds",
                f"round_size not defined in metadata for test id {test_id}",
                traceback.format_stack(),
            )
    if round_pos >= len(df):
        return {}

    df = df[round_pos:round_pos + int(metadata["round_size"])]
    
    return {k:v for k,v in zip(df["new_image_path"].tolist(), df["novel"].tolist())}


class FileProvider(Provider):
    """File-based service provider."""

    def __init__(self, folder: str, results_folder: str):
        """Initialize."""
        self.folder = folder
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def get_test_metadata(self, session_id: str, test_id: str, api_call: bool = True, in_process_only:bool = True) -> Dict[str, Any]:
        """Get test metadata"""
        try:
            structure = get_session_info(self.results_folder, session_id, in_process_only=in_process_only)
            info = structure['created']
            metadata_location = os.path.join(self.folder, info["protocol"], info["domain"], f"{test_id}_metadata.json")
        except KeyError:
            raise ProtocolError("session_id_invalid", f"Provided session id {session_id} could not be found or was improperly set up")

        if not os.path.exists(metadata_location):
            raise ServerError(
                "metadata_not_found",
                f"Metadata file for Test Id {test_id} could not be found",
                traceback.format_stack(),
            )

        hints = []

        # List of metadata vars approved to be sent to the client
        approved_metadata = [
            "protocol",
            "known_classes",
            "max_novel_classes",
            "round_size",
            "feedback_max_ids",
            "pre_novelty_batches",
            "max_detection_feedback_ids"
        ]

        hints = info.get('hints',[])

        approved_metadata.extend([data for data in ["red_light"] if data in hints])

        md = read_meta_data(metadata_location)
        if api_call:
                return {
                    k: v for k, v in md.items() if k in approved_metadata
                }
        return md

    def test_ids_request(
        self, protocol: str, domain: str, detector_seed: str, test_assumptions: str
    ) -> Dict[str, str]:
        def _strip_id(filename):
            return os.path.splitext(os.path.basename(filename))[0]
        """Request test IDs."""
        file_location = os.path.join(self.folder, protocol, domain, "test_ids.csv")
        if not os.path.exists(file_location):
            if not os.path.exists(os.path.join(self.folder, protocol)):
                msg = f"{protocol} not configured"
            elif not os.path.exists(os.path.join(self.folder, protocol, domain)):
                msg = f"domain {domain} for {protocol} not configured"
            else:
                test_ids = [_strip_id(f) for f in glob.glob(os.path.join(self.folder, protocol, domain,'*.csv'))]
                return {"test_ids": test_ids, "generator_seed": "1234"}
            raise ProtocolError(
                "BadDomain",
                msg,
                traceback.format_stack(),
            )

        return {"test_ids": file_location, "generator_seed": "1234"}

    def new_session(
        self, 
        test_ids: List[str], 
        protocol: str, 
        domain: str, 
        novelty_detector_version: str, 
        hints: List[str],
        detection_threshold: float
    ) -> str:
        """Create a session."""
        # Verify's that all given test id's are valid and have associated csv files
        for test_id in test_ids:
            file_location = os.path.join(self.folder, protocol, domain, f"{test_id}_single_df.csv")
            if not os.path.exists(file_location):
                raise ServerError(
                    "test_id_invalid",
                    f"Test Id {test_id} could not be matched to a specific file",
                    traceback.format_stack(),
                )

        session_id = str(uuid.uuid4())

        log_session(
            self.results_folder,
            session_id,
            activity="created",
            content={
                "protocol": protocol,
                "domain": domain,
                "detector": novelty_detector_version,
                "detection_threshold": detection_threshold,
                "hints": hints
            },
        )

        return session_id

    def dataset_request(self, session_id: str, test_id: str, round_id: int) -> FileResult:
        """Request a dataset."""
        try:
            info = get_session_info(self.results_folder, session_id)['created']
            test_info = get_session_test_info(self.results_folder, session_id, test_id)
            file_location = os.path.join(self.folder, info["protocol"], info["domain"], f"{test_id}_single_df.csv")
        except KeyError:
            raise ProtocolError("session_id_invalid", f"Provided session id {session_id} could not be found or was improperly set up")
        
        if not os.path.exists(file_location):
            raise ServerError(
                "test_id_invalid",
                f"Test Id {test_id} could not be matched to a specific file",
                traceback.format_stack(),
            )

        metadata = self.get_test_metadata(session_id, test_id, False)

        if round_id is not None:
            # Check for removing leftover files from restarting tests within a session
            if int(round_id) == 0 and test_info:
                test_session_path = os.path.join(self.results_folder, f"{str(session_id)}.{str(test_id)}.json")
                if os.path.exists(test_session_path):
                    os.remove(test_session_path)
                test_result_paths = glob.glob(os.path.join(
                    self.results_folder, 
                    info["protocol"], 
                    info["domain"], 
                    f"{str(session_id)}.{str(test_id)}_*.csv"
                ))
                for path in test_result_paths:
                    os.remove(path)


            temp_file_path = BytesIO()
            lines = read_gt_csv_file(file_location)
            lines = [x[0] for x in lines if x[0].strip("\n\t\"',.") != ""]
            try:
                    round_pos = int(round_id) * int(metadata["round_size"])
            except KeyError:
                    raise RoundError(
                        "no_defined_rounds",
                        f"round_size not defined in metadata for test id {test_id}",
                        traceback.format_stack(),
                    )
            if round_pos >= len(lines):
                return None

            text = ('\n'.join(lines[round_pos:round_pos + int(metadata["round_size"])]) + "\n").encode('utf-8')
            temp_file_path.write(text)
            temp_file_path.seek(0)
        else:
            temp_file_path = open(file_location, 'rb')

        log_session(
            self.results_folder,
            session_id,
            test_id=test_id,
            round_id=round_id,
            activity="data_request",
        )

        return temp_file_path

    # Sets up the various feedback algorithms that can be used with
    # this implementation of FileProvider
    # {
    #   domain: {
    #       feedback_type: {
    #           function: ...
    #           files: [...]
    #       }
    #   }
    # }
    feedback_request_mapping = {
        "image_classification" : {
            ProtocolConstants.CLASSIFICATION:  {
                "function": get_classification_feedback,
                "files": [ProtocolConstants.CLASSIFICATION ],
                "columns": [1],
                "detection_req": ProtocolConstants.NOTIFY_AND_CONTINUE,
                "budgeted_feedback": True
            },
            ProtocolConstants.SCORE: {
                "function": get_classificaton_score_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [1],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": False
            },
            ProtocolConstants.DETECTION: {
                "function": get_detection_feedback,
                "files": [ProtocolConstants.DETECTION],
                "columns": [0],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True,
                "required_hints": []
            }
        },
        "transcripts" : {
            ProtocolConstants.CLASSIFICATION:  {
                "function": get_classification_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [4],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True
            },
            ProtocolConstants.TRANSCRIPTION: {
                "function": get_levenshtein_feedback,
                "files": [ProtocolConstants.TRANSCRIPTION],
                "columns": [0],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True
            },
            ProtocolConstants.SCORE: {
                "function": get_classificaton_score_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [4],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": False
            }
        },
        "activity_recognition" : {
            ProtocolConstants.CLASSIFICATION:  {
                "function": get_classification_var_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [5, 10],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True
            },
            ProtocolConstants.SCORE: {
                "function": get_classificaton_score_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [2],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": False
            },
            ProtocolConstants.DETECTION: {
                "function": get_detection_feedback,
                "files": [ProtocolConstants.DETECTION],
                "columns": [1],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True,
                "required_hints": []
            }
        }
    }

    def get_feedback(
        self,
        feedback_ids: List[str],
        feedback_type: str,
        session_id: str,
        test_id: str
    ) -> BytesIO:
        """Get feedback of the specified type"""
        metadata = self.get_test_metadata(session_id, test_id, False)
        structure = get_session_info(self.results_folder, session_id)
        test_structure = get_session_test_info(self.results_folder, session_id, test_id)
        domain = structure["created"]["domain"]
        if domain not in self.feedback_request_mapping:
            raise ProtocolError(
                "BadDomain",
                f"The set domain does not match a proper domain type. Please check the metadata file for test {test_id}",
                traceback.format_stack(),
            )


        # Ensure feedback type works with session domain
        # and if so, grab the proper files

        try:
                feedback_definition = self.feedback_request_mapping[domain][feedback_type]
                file_types = feedback_definition["files"]
        except:
                raise ProtocolError(
                    "InvalidFeedbackType",
                    f"Invalid feedback type requested for the test id {test_id} with domain {domain}",
                    traceback.format_stack(),
                )
        is_given_detection_mode = 'red_light' in structure["created"].get('hints', [])
        budgeted_feedback = feedback_definition['budgeted_feedback'] and not \
        (feedback_type ==  ProtocolConstants.DETECTION and is_given_detection_mode)

        if not budgeted_feedback:
            feedback_ids = []

        try:
            # Gets the amount of ids already requested for this type of feedback this round and
            # determines whether the limit has already been reached
            feedback_round_id = str(max([int(r) for r in test_structure["post_results"]["rounds"].keys()]))

            feedback_count = test_structure["get_feedback"]["rounds"][feedback_round_id].get(feedback_type, 0)
            if feedback_count >= metadata["feedback_max_ids"]:
                raise ProtocolError(
                    "FeedbackBudgetExceeded",
                    f"Feedback of type {feedback_type} has already been requested on the maximum number of ids"
                )
        except KeyError:
            feedback_round_id = 0
            feedback_count = 0


        ground_truth_file = os.path.join(self.folder, metadata["protocol"], domain, f"{test_id}_single_df.csv")

        if not os.path.exists(ground_truth_file):
            raise ServerError(
                    "test_id_invalid",
                    f"Could not find ground truth file for test Id {test_id}",
                    traceback.format_stack(),
                )

        results_files = []
        for t in file_types:
                results_files.append(os.path.join(self.results_folder,metadata["protocol"], domain,f"{str(session_id)}.{str(test_id)}_{t}.csv"))

        if len(results_files) < len(file_types):
                raise ServerError(
                    "test_id_invalid",
                    f"Could not find posted result file(s) for test Id {test_id} with feedback type {feedback_type}",
                    traceback.format_stack(),
                )

        # Ensure any required hint(s) are present in the session info structure
        req_hints = feedback_definition.get("required_hints", [])
        if len(req_hints) > 0:
            for hint in req_hints:
                if hint not in structure["created"].get('hints',[]):
                    logging.warning("Inform TA2 team that they are requesting feedback prior to the threshold indication")
                    return BytesIO()

        detection_requirement = feedback_definition.get("detection_req", ProtocolConstants.IGNORE)

        # If novelty detection is required, ensure detection has been posted 
        # for the requested round and novelty claimed for the test
        if detection_requirement != ProtocolConstants.IGNORE:
            test_results_structure = test_structure["post_results"]
            if "detection file path" not in test_results_structure:
                raise ProtocolError(
                    "DetectionPostRequired", 
                    "A detection file is required to be posted before feedback can be requested on a round. Please submit Detection results before requesting feedback"
                )
            else:
                try:
                    with open(test_results_structure["detection file path"], "r") as d_file:
                        d_reader = csv.reader(d_file, delimiter=",")
                        detection_lines = [x for x in d_reader]
                    predictions = [float(x[1]) for x in detection_lines]
                    # if given detection and past the detection point
                    is_given = is_given_detection_mode and metadata.get('red_light') in [x[0] for x in detection_lines]
                    if max(predictions) <= structure["created"]["detection_threshold"] and not is_given:
                        if detection_requirement == ProtocolConstants.NOTIFY_AND_CONTINUE:
                            logging.error("Inform TA2 team that they are requesting feedback prior to the threshold indication")
                        elif detection_requirement == ProtocolConstants.SKIP:
                            logging.warning(
                                "Inform TA2 team that they are requesting feedback prior to the threshold indication")
                            return BytesIO()
                        else:
                            raise ProtocolError(
                             "NoveltyDetectionRequired",
                             f"In order to request {feedback_type} for domain {domain}, novelty must be declared for the test"
                            )
                except ProtocolError as e:
                    raise e
                except Exception as e:
                    raise ServerError(
                        "CantReadFile", 
                        f"Couldnt open the logged detection file at {test_results_structure['detection file path']}. Please check if the file exists and that {session_id}.json has the proper file location for test id {test_id}",
                        traceback.format_exc()
                    )

        # Add columns to metadata for use in feedback
        metadata["columns"] = feedback_definition.get("columns", [])

        # Get feedback from specified test
        try:
            if "psuedo" in feedback_type:
                feedback = psuedo_label_feedback(
                    ground_truth_file,
                    feedback_ids,
                    feedback_definition["files"][0],
                    metadata,
                    self.results_folder,
                    session_id
                )
            else:
                feedback = feedback_definition["function"](
                    ground_truth_file,
                    results_files,
                    feedback_ids,
                    metadata
                )
        except KeyError as e:
            raise ProtocolError(
                "feedback_type_invalid",
                f"Feedback type {feedback_type} is not valid. Make sure the provider's feedback_algorithms variable is properly set up",
                traceback.format_exc()
            )

        number_of_ids_to_return = len(feedback)

        # if budgeted, decrement use and check if too many has been requested
        if budgeted_feedback:
            left_over_ids = int(metadata.get("feedback_max_ids", 0)) - feedback_count
            number_of_ids_to_return = min(number_of_ids_to_return, left_over_ids)
        feedback_count+=number_of_ids_to_return


        log_session(
            self.results_folder,
            session_id=session_id,
            activity="get_feedback",
            test_id=test_id,
            round_id=feedback_round_id,
            content={feedback_type: feedback_count},
        )

        feedback_csv = BytesIO()
        for key in feedback.keys():
            if type(feedback[key]) is not list:
                feedback_csv.write(f"{key},{feedback[key]}\n".encode('utf-8'))
            else:
                feedback_csv.write(f"{key},{','.join(str(x) for x in feedback[key])}\n".encode('utf-8'))
            number_of_ids_to_return-=1
            # once maximium requested number is hit, quit
            if number_of_ids_to_return == 0:
                break

        feedback_csv.seek(0)

        return feedback_csv

    def post_results(
        self,
        session_id: str,
        test_id: str,
        round_id: int,
        result_files: Dict[str, str],
    ) -> None:
        """Post results."""
        # Modify session file with posted results
        structure = get_session_info(self.results_folder, session_id)
        test_structure = get_session_test_info(self.results_folder, session_id, test_id)
        if "detection" in result_files.keys():
            try:
                if "detection" in test_structure["post_results"]["rounds"][str(round_id)]["types"]:
                    raise ProtocolError(
                    "DetectionRepost",
                    "Cannot re post detection for a given round. If you attempted to submit any other results, please resubmit without detection."
                )
            except KeyError:
                pass

        protocol = structure["created"]["protocol"]
        domain = structure["created"]["domain"]
        os.makedirs(os.path.join(self.results_folder, protocol, domain), exist_ok=True)
        log_content = {}
        for r_type in result_files.keys():
            filename = f"{str(session_id)}.{str(test_id)}_{r_type}.csv"
            path = os.path.join(self.results_folder, protocol, domain, filename)
            log_content[f"{r_type} file path"] = path
            with open(path, "a+") as result_file:
                result_file.write(result_files[r_type])

        # Log call
        log_content["last round"] = round_id
        updated_test_structure = log_session(
            self.results_folder,
            activity="post_results",
            session_id=session_id,
            test_id=test_id,
            round_id=round_id,
            content=log_content,
            content_loc="activity",
            return_structure=True
        )

        prev_types = updated_test_structure["post_results"]["rounds"][str(round_id)].get("types", [])
        new_types =  prev_types + list(result_files.keys())
        updated_test_structure["post_results"]["rounds"][str(round_id)]["types"] = new_types
        write_session_log_file(updated_test_structure, os.path.join(self.results_folder, f"{str(session_id)}.{str(test_id)}.json"))

    def evaluate(self, session_id: str, test_id: str, devmode: bool = False) -> Dict:
        """Perform Kitware developed evaluation code modifed to work in this API"""
        from .evaluate.image_classification import ImageClassificationMetrics
        from .evaluate.activity_recognition import ActivityRecognitionMetrics
        from .evaluate.document_transcription import DocumentTranscriptionMetrics
        
        structure = get_session_info(self.results_folder, session_id, in_process_only=False)

        if not devmode:
            if test_id not in structure.get("tests", {}).get("completed_tests", {}):
                raise ProtocolError(
                    "TestInProcess",
                    "The test being evaluated is still in process"
                )
        else:
            structure["created"]["devmode"] = True
            write_session_log_file(structure, os.path.join(self.results_folder, f"{str(session_id)}.json"))
        
        protocol = structure["created"]["protocol"]
        domain = structure["created"]["domain"]
        ground_truth_file = os.path.join(self.folder, protocol, domain, f"{test_id}_single_df.csv")
        gt = pd.read_csv(ground_truth_file, sep=",", header=None, skiprows=1,encoding='utf-8')
        results = {}
        metadata = self.get_test_metadata(session_id, test_id, False, in_process_only=False)

        detection_file_path = os.path.join(
            self.results_folder,
            protocol,
            domain,
            f"{session_id}.{test_id}_detection.csv",
        )
        detections = pd.read_csv(detection_file_path, sep=",", header=None)
        classification_file_path = os.path.join(
            self.results_folder,
            protocol,
            domain,
            f"{session_id}.{test_id}_classification.csv",
        )
        classifications = pd.read_csv(classification_file_path, sep=",", header=None)

        # Image Classification Evaluation
        if domain == "image_classification":
            arm_im = ImageClassificationMetrics(
                protocol, 
                **{"image_id": 0, "detection": 1, "classification": 2}
                )
            m_num = arm_im.m_num(detections[1], gt[arm_im.detection_id])
            results["m_num"] = m_num
            m_num_stats = arm_im.m_num_stats(detections[1], gt[arm_im.detection_id])
            results["m_num_stats"] = m_num_stats
            m_ndp = arm_im.m_ndp(detections[1], gt[arm_im.detection_id])
            results["m_ndp"] = m_ndp
            m_ndp_pre = arm_im.m_ndp_pre(detections[1], gt[arm_im.detection_id])
            results["m_ndp_pre"] = m_ndp_pre
            m_ndp_post = arm_im.m_ndp_post(detections[1], gt[arm_im.detection_id])
            results["m_ndp_post"] = m_ndp_post
            m_acc = arm_im.m_acc(
                gt[arm_im.detection_id],
                classifications,
                gt[arm_im.classification_id],
                100,
                5,
            )
            results["m_acc"] = m_acc
            m_acc_failed = arm_im.m_ndp_failed_reaction(
                detections[arm_im.detection_id],
                gt[1],
                classifications,
                gt[arm_im.classification_id],
            )
            results["m_acc_failed"] = m_acc_failed
            try:
                m_is_cdt_and_is_early = arm_im.m_is_cdt_and_is_early(
                    m_num_stats["GT_indx"], m_num_stats[f"P_indx_{str(metadata['threshold'])}"], gt.shape[0],
                )
            except KeyError:
                m_is_cdt_and_is_early = arm_im.m_is_cdt_and_is_early(
                    m_num_stats["GT_indx"], m_num_stats["P_indx_0.5"], gt.shape[0],
                )
            results["m_is_cdt_and_is_early"] = m_is_cdt_and_is_early

        # Activity Recognition Evaluation
        elif domain == "activity_recognition":
            arm_ar = ActivityRecognitionMetrics(
                protocol, 
                **{
                        "video_id": 0,
                        "novel": 1,
                        "detection": 2,
                        "classification": 3,
                        "spatial": 4,
                        "temporal": 5
                    }
                )
            m_num = arm_ar.m_num(detections[1], gt[arm_ar.novel_id])
            results["m_num"] = m_num
            m_num_stats = arm_ar.m_num_stats(detections[1], gt[arm_ar.novel_id])
            results["m_num_stats"] = m_num_stats
            m_ndp = arm_ar.m_ndp(detections[1], gt[arm_ar.novel_id])
            results["m_ndp"] = m_ndp
            m_ndp_pre = arm_ar.m_ndp_pre(detections[1], gt[arm_ar.novel_id])
            results["m_ndp_pre"] = m_ndp_pre
            m_ndp_post = arm_ar.m_ndp_post(detections[1], gt[arm_ar.novel_id])
            results["m_ndp_post"] = m_ndp_post
            m_acc = arm_ar.m_acc(
                gt[arm_ar.novel_id],
                classifications,
                gt[arm_ar.classification_id],
                100,
                5,
            )
            results["m_acc"] = m_acc
            m_acc_failed = arm_ar.m_ndp_failed_reaction(
                detections[1],
                gt[arm_ar.novel_id],
                classifications,
                gt[arm_ar.classification_id],
            )
            results["m_acc_failed"] = m_acc_failed
            try:
                m_is_cdt_and_is_early = arm_ar.m_is_cdt_and_is_early(
                    m_num_stats["GT_indx"], m_num_stats[f"P_indx_{str(metadata['threshold'])}"], gt.shape[0],
                )
            except KeyError:
                m_is_cdt_and_is_early = arm_ar.m_is_cdt_and_is_early(
                    m_num_stats["GT_indx"], m_num_stats["P_indx_0.5"], gt.shape[0],
                )
            results["m_is_cdt_and_is_early"] = m_is_cdt_and_is_early

        # Document Transcript Evaluation
        elif domain == "transcripts":
            dtm = DocumentTranscriptionMetrics(
                protocol, 
                **{
                        "image_id": 0,
                        "text": 1,
                        "novel": 2,
                        "representation": 3,
                        "detection": 4,
                        "classification": 5,
                        "pen_pressure": 6,
                        "letter_size": 7,
                        "word_spacing": 8,
                        "slant_angle": 9,
                        "attribute": 10
                    }
                )
            m_num = dtm.m_num(detections[1], gt[dtm.novel_id])
            results["m_num"] = m_num
            m_num_stats = dtm.m_num_stats(detections[1], gt[dtm.novel_id])
            results["m_num_stats"] = m_num_stats
            m_ndp = dtm.m_ndp(detections[1], gt[dtm.novel_id])
            results["m_ndp"] = m_ndp
            m_ndp_pre = dtm.m_ndp_pre(detections[1], gt[dtm.novel_id])
            results["m_ndp_pre"] = m_ndp_pre
            m_ndp_post = dtm.m_ndp_post(detections[1], gt[dtm.novel_id])
            results["m_ndp_post"] = m_ndp_post
            m_acc = dtm.m_acc(
                gt[dtm.novel_id], classifications, gt[dtm.classification_id], 100, 5
            )
            results["m_acc"] = m_acc
            m_acc_failed = dtm.m_ndp_failed_reaction(
                detections[1],
                gt[dtm.novel_id],
                classifications,
                gt[dtm.classification_id],
            )
            results["m_acc_failed"] = m_acc_failed
            try:
                m_is_cdt_and_is_early = dtm.m_is_cdt_and_is_early(
                    m_num_stats["GT_indx"], m_num_stats[f"P_indx_{str(metadata['threshold'])}"], gt.shape[0],
                )
            except KeyError:
                m_is_cdt_and_is_early = dtm.m_is_cdt_and_is_early(
                    m_num_stats["GT_indx"], m_num_stats["P_indx_0.3"], gt.shape[0],
                )
            results["m_is_cdt_and_is_early"] = m_is_cdt_and_is_early
        else:
            raise ProtocolError(
                "BadDomain",
                f"Domain {domain} retrieved from session log for {session_id} does not match a proper domain type",
                traceback.format_stack(),
            )


        # Metrics functions return ints as int64's which are 
        # not compatible with json and must be converted
        for k in results.keys():
            for key in results[k].keys():
                if type(results[k][key]) == np.int64:
                    results[k][key] = int(results[k][key])
        return results

    def complete_test(self, session_id: str, test_id: str) -> None:
        """Mark test as completed in session logs"""
        log_session(self.results_folder, session_id=session_id, test_id=test_id, activity="completion")
        
    def terminate_session(self, session_id: str) -> None:
        """Terminate the session."""
        # Modify session file to indicate session has been terminated
        log_session(self.results_folder, session_id=session_id, activity="termination")

    def session_status(self, after: str = None, session_id: str = None, include_tests: bool = False,
                       test_ids:List[str] = None, detector: str = None) -> str:
        """
        Retrieve Session Names
        :param after: Date Time String lower
        :param include_tests if True, then add the completed test
        :return: CSV of session id and start date time and, termination date time stamp in iso format
        if include tests, then add a second column of tests, thus the format is:
           session_id, test_id, start date time, termination date time stamp
        """
        lower_bound = parser.isoparse(after) if after is not None else None

        session_files_locations = glob.glob(
            os.path.join(os.path.join(self.results_folder, "*.json"))
        )

        results = []
        for session_file in session_files_locations:
            with open(session_file, 'r') as fp:
                session_data = json.load(fp)
            terminated = 'termination' in session_data
            created = 'created' in session_data
            if terminated:
                terminate_time = session_data['termination']['time'][0]
            else:
                terminate_time = 'Incomplete'
            if created:
                creation_time = session_data['created']['time'][0]
            else:
                creation_time = 'N/A'
            session_detector = session_data['created']['detector'] if created else None
            session_name = os.path.splitext(os.path.basename(session_file))[0]
            tests = session_data.get('tests', {}).get('completed_tests', {}) if include_tests and created else None
            if detector is not None and detector != session_detector:
                continue
            if (session_id is None and (not lower_bound or lower_bound <= parser.isoparse(terminate_time))) or session_name == session_id:
                if include_tests:
                    if test_ids is None:
                        test_ids = tests
                    if tests and test_ids:
                        for test_id in test_ids:
                            if test_id in tests:
                                session_test_file = f"{session_file[:-5]}.{test_id}.json"
                                with open(session_test_file, "r") as tf:
                                    test_data = json.load(tf)
                                try:
                                    creation_time = test_data['post_results']['rounds']['0']['time'][0]
                                except KeyError:
                                    creation_time = "N/A"
                            else:
                                creation_time = 'N/A'
                            results.append(f'{session_name},{session_detector},{test_id},{creation_time},{terminate_time}')
                    else:
                        results.append(f'{session_name},{session_detector}, NA,{creation_time},{terminate_time}')
                else:
                    results.append(f'{session_name},{session_detector},{creation_time},{terminate_time}')
        results = sorted(results, key=lambda x: (x.split(',')[1], x.split(',')[0]))
        return '\n'.join(results)

    def get_session_zip(self, session_id: str, test_ids: List[str] = None) -> str:
        """
        Retrieve Completed Session Names
        :param session_id
        :param test_id
        :return: zip file path
        """
        zip_file_name = os.path.join(self.results_folder, f'{session_id}.zip')
        with zipfile.ZipFile(zip_file_name, 'w', compression= zipfile.ZIP_BZIP2) as zip:
            zip.write(os.path.join(self.results_folder, f'{session_id}.json'), arcname=f'{session_id}.json')

            for protocol in os.listdir(self.results_folder):
                if os.path.isdir(os.path.join(self.results_folder, protocol)):
                    if test_ids is None:
                        test_files = glob.glob(
                            os.path.join(self.results_folder, protocol, "**", f"{session_id}.*.csv"),
                            recursive=True,
                        )
                        test_files.extend(glob.glob(
                            os.path.join(self.results_folder, f"{session_id}.*.json")
                        ))
                    else:
                        test_files = []
                        for test_id in test_ids:
                            test_files.extend(glob.glob(
                                os.path.join(self.results_folder, protocol, "**", f"{session_id}.{test_id}*.csv"),
                                recursive=True,
                            ))
                            test_files.extend(os.path.join(self.results_folder, f"{session_id}.{test_id}.json"))
                    for test_file in test_files:
                        zip.write(test_file, arcname=test_file[len(self.results_folder) + 1:])

        return zip_file_name

    def latest_session_info(self, session_id: str) -> str:
        structure = get_session_info(self.results_folder, session_id)
        latest = {}
        latest["finished_tests"] = structure["tests"]["completed_tests"]
        return latest

class FileProviderSVO(FileProvider):
    def __init__(self, folder: str, results_folder: str):
        super().__init__(folder, results_folder)
    
    feedback_request_mapping = { "svo_classification" : 
                                    {
                                        ProtocolConstants.CLASSIFICATION:  {
                                            "function": get_classification_feedback_topk,
                                            "files": [ProtocolConstants.CLASSIFICATION ],
                                            "columns": {"subject" : [3], "verb" : [9], "object" : [6]}, # HARDCODE: Index of S,V,O in ground truth
                                            "detection_req": ProtocolConstants.NOTIFY_AND_CONTINUE,
                                            "budgeted_feedback": True
                                        },
                                        ProtocolConstants.DETECTION: {
                                            "function": get_detection_feedback,
                                            "files": [ProtocolConstants.DETECTION],
                                            "columns": [-1], # HARDCODE
                                            "detection_req": ProtocolConstants.SKIP,
                                            "budgeted_feedback": True,
                                            "required_hints": [],
                                            "alternate_budget": "max_detection_feedback_ids"
                                        }
                                    }
                                }

    hint_request_mapping = { "svo_classification" : 
                                {
                                    ProtocolConstants.TYPE_A:  {
                                        "function": get_hint_typeA,
                                        "files": [ProtocolConstants.TYPE_A],
                                        "columns": [1] # HARDCODE
                                    },
                                    ProtocolConstants.TYPE_B: {
                                        "function": get_hint_typeB,
                                        "files": [ProtocolConstants.TYPE_A],
                                        "columns": [-1], # HARDCODE
                                    }
                                }
                            }

    def dataset_request(self, session_id: str, test_id: str, round_id: int) -> FileResult:
        """Request a dataset."""
        try:
            info = get_session_info(self.results_folder, session_id)['created']
            test_info = get_session_test_info(self.results_folder, session_id, test_id)
            file_location = os.path.join(self.folder, info["protocol"], info["domain"], f"{test_id}_single_df.csv")
        except KeyError:
            raise ProtocolError("session_id_invalid", f"Provided session id {session_id} could not be found or was improperly set up")
        
        if not os.path.exists(file_location):
            raise ServerError(
                "test_id_invalid",
                f"Test Id {test_id} could not be matched to a specific file",
                traceback.format_stack(),
            )

        metadata = self.get_test_metadata(session_id, test_id, False)

        if round_id is not None:
            # Check for removing leftover files from restarting tests within a session
            if int(round_id) == 0 and test_info:
                test_session_path = os.path.join(self.results_folder, f"{str(session_id)}.{str(test_id)}.json")
                if os.path.exists(test_session_path):
                    os.remove(test_session_path)
                test_result_paths = glob.glob(os.path.join(
                    self.results_folder, 
                    info["protocol"], 
                    info["domain"], 
                    f"{str(session_id)}.{str(test_id)}_*.csv"
                ))
                for path in test_result_paths:
                    os.remove(path)


            temp_file_path = BytesIO()
            lines = read_gt_csv_file(file_location)
            # HARDCODED 
            lines = [[x[0],x[-9:-1]] for x in lines if x[0].strip("\n\t\"',.") != ""]
            try:
                    round_pos = int(round_id) * int(metadata["round_size"])
            except KeyError:
                    raise RoundError(
                        "no_defined_rounds",
                        f"round_size not defined in metadata for test id {test_id}",
                        traceback.format_stack(),
                    )
            if round_pos >= len(lines):
                return None

            # text = ('\n'.join(lines[round_pos:round_pos + int(metadata["round_size"])]) + "\n").encode('utf-8')
            text = str(lines[round_pos:round_pos + int(metadata["round_size"])]).encode('utf-8')
            temp_file_path.write(text)
            temp_file_path.seek(0)
        else:
            temp_file_path = open(file_location, 'rb')

        log_session(
            self.results_folder,
            session_id,
            test_id=test_id,
            round_id=round_id,
            activity="data_request",
        )

        return temp_file_path

    def get_feedback(
        self,
        feedback_ids: List[str],
        feedback_type: str,
        session_id: str,
        test_id: str,
        feedback_category: str
    ) -> BytesIO:
        """Get feedback of the specified type"""
        metadata = self.get_test_metadata(session_id, test_id, False)
        structure = get_session_info(self.results_folder, session_id)
        test_structure = get_session_test_info(self.results_folder, session_id, test_id)
        domain = structure["created"]["domain"]
        if domain not in self.feedback_request_mapping:
            raise ProtocolError(
                "BadDomain",
                f"The set domain does not match a proper domain type. Please check the metadata file for test {test_id}",
                traceback.format_stack(),
            )


        # Ensure feedback type works with session domain
        # and if so, grab the proper files

        try:
                feedback_definition = self.feedback_request_mapping[domain][feedback_type]
                file_types = feedback_definition["files"]
        except:
                raise ProtocolError(
                    "InvalidFeedbackType",
                    f"Invalid feedback type requested for the test id {test_id} with domain {domain}",
                    traceback.format_stack(),
                )
        is_given_detection_mode = 'red_light' in structure["created"].get('hints', [])
        budgeted_feedback = feedback_definition['budgeted_feedback'] and not \
        (feedback_type ==  ProtocolConstants.DETECTION and is_given_detection_mode)

        if "alternate_budget" in  feedback_definition and feedback_definition["alternate_budget"] in metadata:
            feedback_budget = int(metadata[feedback_definition["alternate_budget"]])
        else:
            feedback_budget = int(metadata.get("feedback_max_ids",0))

        if not budgeted_feedback:
            feedback_ids = []

        # Gets the amount of ids already requested for this type of feedback this round and
        # determines whether the limit has already been reached
        feedback_round_id = str(max([int(r) for r in test_structure["post_results"]["rounds"].keys()]))
        
        try:
            feedback_count = test_structure["get_feedback"]["rounds"][feedback_round_id].get(feedback_type, 0)
            if feedback_count >= feedback_budget:
                raise ProtocolError(
                    "FeedbackBudgetExceeded",
                    f"Feedback of type {feedback_type} has already been requested on the maximum number of ids"
                )
        except KeyError:
            feedback_count = 0
            # print(f"FC : {feedback_count}, FCAT : {feedback_category}, FR : {feedback_round_id}")


        ground_truth_file = os.path.join(self.folder, metadata["protocol"], domain, f"{test_id}_single_df.csv")

        if not os.path.exists(ground_truth_file):
            raise ServerError(
                    "test_id_invalid",
                    f"Could not find ground truth file for test Id {test_id}",
                    traceback.format_stack(),
                )

        results_files = []
        for t in file_types:
                results_files.append(os.path.join(self.results_folder,metadata["protocol"], domain,f"{str(session_id)}.{str(test_id)}_{t}.csv"))

        if len(results_files) < len(file_types):
                raise ServerError(
                    "test_id_invalid",
                    f"Could not find posted result file(s) for test Id {test_id} with feedback type {feedback_type}",
                    traceback.format_stack(),
                )

        # Ensure any required hint(s) are present in the session info structure
        req_hints = feedback_definition.get("required_hints", [])
        if len(req_hints) > 0:
            for hint in req_hints:
                if hint not in structure["created"].get('hints',[]):
                    logging.warning("Inform TA2 team that they are requesting feedback prior to the threshold indication")
                    return BytesIO()

        detection_requirement = feedback_definition.get("detection_req", ProtocolConstants.IGNORE)

        # If novelty detection is required, ensure detection has been posted 
        # for the requested round and novelty claimed for the test
        if detection_requirement != ProtocolConstants.IGNORE:
            test_results_structure = test_structure["post_results"]
            if "detection file path" not in test_results_structure:
                raise ProtocolError(
                    "DetectionPostRequired", 
                    "A detection file is required to be posted before feedback can be requested on a round. Please submit Detection results before requesting feedback"
                )
            else:
                try:
                    with open(test_results_structure["detection file path"], "r") as d_file:
                        d_reader = csv.reader(d_file, delimiter=",")
                        # TODO : confirm this once
                        detection_lines = [x for x in d_reader][1:]
                    predictions = [float(x[1]) for x in detection_lines]
                    # if given detection and past the detection point
                    is_given = is_given_detection_mode and metadata.get('red_light') in [x[0] for x in detection_lines]
                    if max(predictions) <= structure["created"]["detection_threshold"] and not is_given:
                        if detection_requirement == ProtocolConstants.NOTIFY_AND_CONTINUE:
                            logging.error("Inform TA2 team that they are requesting feedback prior to the threshold indication")
                        elif detection_requirement == ProtocolConstants.SKIP:
                            logging.warning(
                                "Inform TA2 team that they are requesting feedback prior to the threshold indication")
                            return BytesIO()
                        else:
                            raise ProtocolError(
                             "NoveltyDetectionRequired",
                             f"In order to request {feedback_type} for domain {domain}, novelty must be declared for the test"
                            )
                except ProtocolError as e:
                    raise e
                except Exception as e:
                    raise ServerError(
                        "CantReadFile", 
                        f"Couldnt open the logged detection file at {test_results_structure['detection file path']}. Please check if the file exists and that {session_id}.json has the proper file location for test id {test_id}",
                        traceback.format_exc()
                    )

        # Add columns to metadata for use in feedback
        metadata["columns"] = feedback_definition.get("columns", [])

        topk_file = os.path.join(self.folder, metadata["protocol"], domain, f"{feedback_category}_topk.csv")

        if not os.path.isfile(topk_file) and feedback_type == "classification":
            raise ServerError(
                    "topk file invalid",
                    f"Could not find ground truth file for Category {feedback_category}",
                    traceback.format_stack(),
                )

        # Get feedback from specified test
        try:
            if "psuedo" in feedback_type:
                feedback = psuedo_label_feedback(
                    ground_truth_file,
                    feedback_ids,
                    feedback_definition["files"][0],
                    metadata,
                    self.results_folder,
                    session_id
                )
            elif "classification" in feedback_type:
                metadata["columns"] = metadata["columns"][feedback_category]
                feedback = feedback_definition["function"](
                    ground_truth_file,
                    topk_file,
                    results_files,
                    feedback_ids,
                    metadata
                )
            else:
                feedback = feedback_definition["function"](
                    ground_truth_file,
                    results_files,
                    feedback_ids,
                    metadata,
                    is_given_detection_mode
                )
        except KeyError as e:
            raise ProtocolError(
                "feedback_type_invalid",
                f"Feedback type {feedback_type} is not valid. Make sure the provider's feedback_algorithms variable is properly set up",
                traceback.format_exc()
            )

        number_of_ids_to_return = len(feedback)

        # if budgeted, decrement use and check if too many has been requested
        if budgeted_feedback:
            left_over_ids = feedback_budget - feedback_count
            number_of_ids_to_return = min(number_of_ids_to_return, left_over_ids)
        feedback_count+=number_of_ids_to_return


        log_session(
            self.results_folder,
            session_id=session_id,
            activity="get_feedback",
            test_id=test_id,
            round_id=feedback_round_id,
            content={feedback_type: feedback_count},
        )

        feedback_csv = BytesIO()
        for key in feedback.keys():
            if type(feedback[key]) is not list:
                feedback_csv.write(f"{key},{feedback[key]}\n".encode('utf-8'))
            else:
                feedback_csv.write(f"{key},{','.join(str(x) for x in feedback[key])}\n".encode('utf-8'))
            number_of_ids_to_return-=1
            # once maximium requested number is hit, quit
            if number_of_ids_to_return == 0:
                break

        feedback_csv.seek(0)

        return feedback_csv
    
    def get_hint(
        self,
        session_id: str,
        test_id: str,
        round_id: str,
        hint_type: str
    ) -> BytesIO:
        """Get Hint of the specified type"""

        metadata = self.get_test_metadata(session_id, test_id, False)
        structure = get_session_info(self.results_folder, session_id)
        test_structure = get_session_test_info(self.results_folder, session_id, test_id)
        domain = structure["created"]["domain"]
        if domain not in self.hint_request_mapping:
            raise ProtocolError(
                "BadDomain",
                f"The set domain does not match a proper domain type. Please check the metadata file for test {test_id}",
                traceback.format_stack(),
            )


        try:
                hint_definition = self.hint_request_mapping[domain][hint_type]
                file_types = hint_definition["files"]
        except:
                raise ProtocolError(
                    "InvalidFeedbackType",
                    f"Invalid feedback type requested for the test id {test_id} with domain {domain}",
                    traceback.format_stack(),
                )
        
        # if hints is 'red_light' means it is given detection mode
        is_given_detection_mode = 'red_light' in structure["created"].get('hints', [])

        ground_truth_file = os.path.join(self.folder, metadata["protocol"], domain, f"{test_id}_single_df.csv")

        if not os.path.exists(ground_truth_file):
            raise ServerError(
                    "test_id_invalid",
                    f"Could not find ground truth file for test Id {test_id}",
                    traceback.format_stack(),
                )

        # results_files = []
        # for t in file_types:
        #         results_files.append(os.path.join(self.results_folder,metadata["protocol"], domain,f"{str(session_id)}.{str(test_id)}_{t}.csv"))

        # if len(results_files) < len(file_types):
        #         raise ServerError(
        #             "test_id_invalid",
        #             f"Could not find posted result file(s) for test Id {test_id} with feedback type {feedback_type}",
        #             traceback.format_stack(),
        #         )

        # Add columns to metadata for use in feedback
        metadata["columns"] = hint_definition.get("columns", [])
        metadata["curr_test_id"] = test_id
        metadata["curr_round_id"] = round_id

        hint_file = os.path.join(self.folder, metadata["protocol"], domain, f"hints_info.csv")

        if not os.path.isfile(hint_file):
            raise ServerError(
                    "Hint file invalid",
                    "Could not find ground truth file for ",
                    traceback.format_stack(),
                )

        if not is_given_detection_mode and hint_type == ProtocolConstants.TYPE_B:
            raise ServerError(
                    "Invalid Hint type",
                    "Type B hint not supported by system detection",
                    traceback.format_stack(),
                )
        try:
            hint = hint_definition["function"](
                ground_truth_file,
                metadata,
                hint_file,
            )
        except KeyError as e:
            raise ProtocolError(
                "Hint_type_invalid",
                f"Hint type {hint_type} is not valid. Make sure the provider's hint_algorithms variable is properly set up",
                traceback.format_exc()
            )


        log_session(
            self.results_folder,
            session_id=session_id,
            activity="get_hint",
            test_id=test_id,
            round_id=round_id,
        )

        hint_csv = BytesIO()
        for key in hint.keys():
            if type(hint[key]) is not list:
                hint_csv.write(f"{key},{hint[key]}\n".encode('utf-8'))
            else:
                hint_csv.write(f"{key},{','.join(str(x) for x in hint[key])}\n".encode('utf-8'))

        hint_csv.seek(0)

        return hint_csv

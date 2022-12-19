"""Definition of the Provider interface."""

from abc import ABC, abstractmethod

from typing import List, Dict, Any, IO
from io import BytesIO

FileResult = IO

"""
File name and indication of temporary requiring clenaup
"""


class Provider(ABC):
    """
    Abstract base class for serverside service provider functionality.

    Provider is an abstract class that is intended to be inherited, with custom
    implementations, and passed onto the API.
    """

    @abstractmethod
    def get_test_metadata(self, session_id: str, test_id: str, api_call: bool = True) -> Dict[str, Any]:
        """Get test metadata"""
        pass

    @abstractmethod
    def test_ids_request(
        self, protocol: str, domain: str, detector_seed: str, test_assumptions: str
    ) -> Dict[str, str]:
        """Request test IDs."""
        pass

    @abstractmethod
    def new_session(
        self, 
        test_ids: List[str], 
        protocol: str, 
        novelty_detector_version: str, 
        hints: List[str], 
        detection_threshold: float
    ) -> str:
        """Create a new session."""
        pass

    @abstractmethod
    def dataset_request(self, session_id: str, test_id: str, round_id: int) -> FileResult:
        """Request a dataset."""
        pass

    """
        Dictionary of algorithms used for storing functions
        for the various types of feedback.
        All function implementations need to take as params the following:
        -Ground truth file csv reader: reader
        -Result file csv reader: reader
        -Feedback ids: List[str]
        -Metadata: Dict[str, Any]
    """
    feedback_algorithms = {}

    @abstractmethod
    def get_feedback(
        self,
        feedback_ids: List[str],
        feedback_type: str,
        session_id: str,
        test_id: str,
        round_id: int,
    ) -> BytesIO:
        """Get feedback."""
        pass

    @abstractmethod
    def post_results(
        self,
        session_id: str,
        test_id: str,
        round_id: int,
        result_files: Dict[str, str],
    ) -> None:
        """Post results."""
        pass

    @abstractmethod
    def evaluate(self, session_id: str, test_id: str) -> str:
        """Perform Kitware developed evaluation code modifed to work in this API"""
        pass

    @abstractmethod
    def terminate_session(self, session_id: str) -> None:
        """Terminate a session."""
        pass
    
    @abstractmethod
    def session_status(
        self, 
        after: str = None, 
        session_id: str = None, 
        include_tests: bool = False, 
        test_ids:List[str] = None,
        detector: str = None
    ) -> str:
        """Gets the session status"""
        pass

    @abstractmethod
    def get_session_zip(self, session_id: str, test_ids: List[str] = None) -> str:
        """Returns a zip of all files assoiated with the provided session id and any provided test ids"""
        pass

    @abstractmethod
    def latest_session_info(self, session_id: str) -> str:
        """Return a dict with a list of all completed tests"""
        pass

    @abstractmethod
    def complete_test(self, session_id: str, test_id: str) -> None:
        """Mark the given test as complete"""
        pass

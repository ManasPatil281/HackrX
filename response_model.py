# Define request and response models
from typing import List, Union
from pydantic import BaseModel


class DebugRequest(BaseModel):
    question: str

class QueryRequest(BaseModel):
    documents: Union[List[str], str]  # Allow both list of strings and single string
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]
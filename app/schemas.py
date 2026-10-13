"""Request and response models. The request is the API's data contract:
it mirrors the Pandera schema used in training (src/data.py)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNoUnknown = Literal["no", "yes", "unknown"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(ge=17, le=100)
    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    ]
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown
    contact: Literal["cellular", "telephone"]
    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
    ]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0, le=999, description="days since last contact; 999 = never contacted")
    previous: int = Field(ge=0)
    poutcome: Literal["failure", "nonexistent", "success"]
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")


class PredictionResponse(BaseModel):
    request_id: str
    probability: float
    prediction: int
    model_version: str


class FeedbackRequest(BaseModel):
    request_id: str
    actual_outcome: int = Field(ge=0, le=1)

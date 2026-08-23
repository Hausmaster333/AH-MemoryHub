from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import math
import uuid


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SensoryRepresentation(Frozen):
    modality: str = Field(min_length=1)
    value: str = Field(min_length=1)


class FirstOrderSymbol(Frozen):
    uid: str = Field(min_length=1)
    sensory_representations: tuple[SensoryRepresentation, ...] = Field(min_length=1)

    @field_validator("sensory_representations")
    @classmethod
    def distinct_modalities(cls, v: tuple[SensoryRepresentation, ...]):
        if len({(x.modality, x.value) for x in v}) != len(v):
            raise ValueError("duplicate sensory representation")
        return v


class SReference(Frozen):
    kind: Literal["S"] = "S"
    reference_uid: str = Field(min_length=1)
    target_uid: str = Field(min_length=1)


class MReference(Frozen):
    kind: Literal["M"] = "M"
    reference_uid: str = Field(min_length=1)
    target_uid: str = Field(min_length=1)


class ElementReference(Frozen):
    kind: Literal["E"] = "E"
    reference_uid: str = Field(min_length=1)
    target_uid: str = Field(min_length=1)


Reference = SReference | MReference | ElementReference


class Property(Frozen):
    name: str = Field(min_length=1)
    value: Any
    value_type: str = "string"
    unit: str | None = None


class SourceEvidence(Frozen):
    document_uid: str
    chunk_uid: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    exact_text: str
    content_hash: str
    parser_run_uid: str
    model_id: str = "rule-based-demo"
    parser_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def span_order(self):
        if self.end_offset < self.start_offset:
            raise ValueError("invalid source span")
        return self


class SecondOrderSymbol(Frozen):
    uid: str
    properties: tuple[Property, ...] = ()
    meta_properties: tuple[Property, ...] = ()
    evidence: tuple[SourceEvidence, ...] = ()


class FunctionalSymbol(Frozen):
    uid: str
    function_id: str
    ordered_operands: tuple[Reference, ...] = ()


class MemoryList(Frozen):
    uid: str
    ordered_members: tuple[Reference, ...] = ()
    list_type: str = "generic"
    properties: tuple[Property, ...] = ()
    meta_properties: tuple[Property, ...] = ()


class Role(Frozen):
    role_id: str
    required: bool = False
    multiplicity: bool = False


class ControlTemplate(Frozen):
    uid: str
    predicate_ref: SReference
    ordered_roles: tuple[Role, ...]


class RoleBinding(Frozen):
    role_id: str
    target_ref: Reference


class Hypernode(Frozen):
    uid: str
    weight: float = Field(ge=0, le=1)
    template_ref: ElementReference
    role_bindings: tuple[RoleBinding, ...] = ()
    properties: tuple[Property, ...] = ()
    meta_properties: tuple[Property, ...] = ()
    evidence: tuple[SourceEvidence, ...] = ()
    created_tick: int = 0
    origin: Literal["manual", "organiser", "ingestion"] = "manual"

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, v: float):
        if not math.isfinite(v):
            raise ValueError("weight must be finite")
        return v


class AssociativeLink(Frozen):
    uid: str
    type_id: str
    weight: float = Field(ge=0, le=1)
    source_ref: Reference
    target_ref: Reference

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, v: float):
        if not math.isfinite(v):
            raise ValueError("weight must be finite")
        return v


Payload = SReference | SecondOrderSymbol | MReference | FunctionalSymbol | MemoryList | ControlTemplate | Hypernode


class MemoryElement(Frozen):
    uid: str
    payload: Payload
    excitation: float = Field(default=0.0, ge=0, le=1)
    activation_function: str = "clip_sum"

    @model_validator(mode="after")
    def payload_identity(self):
        payload_uid = self.payload.reference_uid if isinstance(self.payload, (SReference, MReference)) else self.payload.uid
        if self.uid != payload_uid:
            raise ValueError("MemoryElement UID must match payload identity")
        return self


class CandidateBinding(Frozen):
    role_id: str
    value: str = Field(min_length=1)
    observed: str | None = None


class CandidateFact(Frozen):
    predicate: str = Field(min_length=1)
    bindings: tuple[CandidateBinding, ...] = Field(min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    exact_text: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    unresolved_entities: bool = False
    model_id: str = "rule-based-demo"
    span_uid: str | None = None
    sentence_index: int = Field(default=0, ge=0)
    context_before: str = ""
    context_after: str = ""

    @model_validator(mode="after")
    def valid_span(self):
        if self.source_end < self.source_start:
            raise ValueError("invalid source span")
        return self


class DocumentIngestRequest(BaseModel):
    text: str = Field(min_length=1)
    document_uid: str | None = None
    source_name: str = "api"


class CandidateDecisionRequest(BaseModel):
    preview_uid: str = Field(min_length=1)
    candidate_uid: str = Field(min_length=1)
    decision: Literal["admit", "reject"]


class FactIngestRequest(BaseModel):
    text: str = Field(min_length=1)
    document_uid: str | None = None


class IgnitionConfig(Frozen):
    initial_life_ticks: int = Field(default=5, ge=0)
    decay_lambda: float = Field(default=0.08, ge=0)
    working_memory_threshold: float = Field(default=0.2, ge=0, le=1)
    hebbian_eta: float = Field(default=0.0, ge=0, le=1)
    rhythm_hz: float = Field(default=1.0, gt=0)
    max_ticks: int = Field(default=8, ge=1, le=1000)
    epsilon: float = Field(default=1e-4, gt=0)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    max_ticks: int = Field(default=8, ge=1, le=100)
    profile: Literal["literal_2026", "integrated_v1"] = "integrated_v1"
    ignition: IgnitionConfig | None = None

    @model_validator(mode="after")
    def apply_max_ticks(self):
        if self.ignition is not None and self.max_ticks != 8:
            self.ignition = self.ignition.model_copy(update={"max_ticks": self.max_ticks})
        return self


class DSLRequest(BaseModel):
    expression: str = Field(min_length=1, max_length=4000)


class GCCommitRequest(BaseModel):
    preview_token: str = Field(min_length=1)
    deletable_uids: list[str] | None = None
    uids: list[str] | None = None

    @model_validator(mode="after")
    def normalized_uids(self):
        if self.deletable_uids is None and self.uids is not None: self.deletable_uids = self.uids
        return self


class EvaluationRequest(BaseModel):
    fixture: str = "internal"
    gold: list[dict[str, Any]] = []
    predicted: list[dict[str, Any]] = []


class TickTrace(Frozen):
    tick: int
    source_uid: str | None = None
    target_uid: str
    link_or_hypernode_uid: str | None = None
    impulse_type: str
    impulse_value: float
    previous_excitation: float
    next_excitation: float
    activation: float
    threshold_entry: bool = False
    threshold_exit: bool = False
    previous_weight: float | None = None
    next_weight: float | None = None
    parent_trace: int | None = None


class IgnitionRun(Frozen):
    run_uid: str
    profile: str
    source_revision: int
    ticks: tuple[TickTrace, ...]
    working_memory: tuple[str, ...]
    status: str
    evidence_uids: tuple[str, ...] = ()
    effective_config: IgnitionConfig = IgnitionConfig()
    minimal_path: tuple[str, ...] = ()
    trace_complete: bool = False
    weight_deltas: dict[str, float] = {}

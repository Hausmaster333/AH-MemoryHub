# AH-MemoryHub

AH-MemoryHub turns natural-language experience into a typed associative-heterarchical memory and exposes how that memory reaches an answer.

## Memory structure

**AH Memory**:
The complete memory tuple `AH = <S, C, P, H, L>` owned by one cognitive agent.
_Avoid_: Knowledge graph, graph database

**First-order Symbol**:
A grounded abstract symbol in `S` with a unique identity and modality-labelled sensory representations.
_Avoid_: Entity, token

**Sensory Representation**:
A primary symbol in one sensory modality that grounds a First-order Symbol; the hackathon profile initially uses the text modality.
_Avoid_: Embedding, chunk

**Memory Element**:
A typed member of exactly one of the `C`, `P`, or `H` sections, carrying excitation state and an activation function.
_Avoid_: Node, record

**Common Knowledge**:
The `C` section containing knowledge shareable between agents.
_Avoid_: Public graph

**Private Knowledge**:
The `P` section containing agent-specific knowledge that is not a personal-history episode.
_Avoid_: User data

**Personal History**:
The `H` section containing facts and episodes experienced or attributed to the agent.
_Avoid_: Log, chat history

**S Reference**:
An independently identified `s*` pointer whose target is a First-order Symbol in `S`.
_Avoid_: Foreign key, edge, generic reference

**M Reference**:
An independently identified `m*` pointer whose target is specifically a Second-order Symbol.
_Avoid_: Memory Element reference, generic reference

**Element Reference**:
An independently identified `e*` pointer whose target is any Memory Element in `C`, `P`, or `H`; a template reference `t*` is an Element Reference constrained to a Control Template.
_Avoid_: M Reference, ordinary edge

**Second-order Symbol**:
A symbolic Memory Element whose meaning is described by properties, meta-properties, references, and relations.
_Avoid_: Entity

**Functional Symbol**:
A Memory Element whose identifier selects agent behaviour for its ordered operands.
_Avoid_: Function call

**Memory List**:
An identified ordered collection of references used to represent a hierarchy, heterarchy, episode, or object model.
_Avoid_: Set, array

**Control Template**:
A predicate model declaring the roles that a corresponding fact may fill.
_Avoid_: Prompt template, schema

**Role**:
A semantic vacancy in a Control Template, such as SUBJECT, OBJECT, LOCATION, TIME, CAUSE, or TOOL.
_Avoid_: Edge label

**Role Binding**:
The explicit pairing of one Role with one referenced actant in a fact.
_Avoid_: Argument position

**Hypernode**:
An identified, weighted n-ary fact that instantiates a Control Template through Role Bindings.
_Avoid_: Relation, ordinary edge

**Associative Link**:
An identified, typed, weighted, directed connection between addressable parts of AH Memory.
_Avoid_: Hypernode, relationship

## Derived structures

**Hierarchy**:
A Memory List whose members form a directed acyclic graph through one homogeneous Associative Link type, such as IS-A.

**Heterarchy**:
A Memory List whose members form a connected subgraph through heterogeneous Associative Link types.

**Episode**:
A Memory List in Personal History whose facts form a directed acyclic graph through FOLLOW links.
_Avoid_: Conversation, session

## Inference

**Candidate Fact**:
A probabilistic extraction proposed from source text before deterministic validation and admission to AH Memory.
_Avoid_: Fact

**Ignition Run**:
One bounded, tick-based propagation over an immutable AH Memory snapshot.
_Avoid_: Search, graph traversal

**Excitation**:
The internal dynamic state `x` of a Memory Element during an Ignition Run.
_Avoid_: Confidence, truth score

**Activation**:
The output of an element's activation function for the current tick.
_Avoid_: Excitation

**Working Memory**:
The elements whose excitation exceeds the configured threshold for the current tick.
_Avoid_: Cache, context window

**Ignition Trace**:
The ordered evidence of impulses, activations, threshold crossings, and weight changes that actually contributed to an answer.
_Avoid_: Debug log, chain of thought

**Source Evidence**:
An immutable reference from an admitted memory fact to its document and exact source span.
_Avoid_: Citation string

**Query Seed**:
A grounded addressable memory item whose meaning is explicitly present in the question and from which an Ignition Run begins.
_Avoid_: Search term, every retrieved fact

**Candidate Subgraph**:
A bounded retrieval projection that may contain relevant symbols, facts, and links but is not itself evidence for an answer.
_Avoid_: Working Memory, Evidence Path

**Evidence Path**:
The trace-supported sequence from Query Seeds through activated facts and actants that directly supports an answer.
_Avoid_: Candidate Subgraph, all trace events

**Answer Evidence**:
The Source Evidence attached to facts on the Evidence Path and explicitly cited by the final answer.
_Avoid_: All retrieved sources, prompt context

## Perception

**Source Span**:
An exact, offset-addressed fragment of the untouched input document with optional neighbouring context.
_Avoid_: Sentence string, chunk

**Mention**:
An occurrence of an entity, event, state, time, or location anchored to a Source Span; several Mentions may denote the same referent.
_Avoid_: Entity, Symbol

**Coreference**:
A directed assertion that one Mention denotes the same referent as an earlier Mention.
_Avoid_: String replacement, pronoun rule

**Candidate Term**:
A proposed actant value that is either one Mention or an ordered composition of Candidate Terms through `AND`, `OR`, or `VERY`.
_Avoid_: Binding string, final Symbol

**Candidate Fact**:
One atomic, source-grounded predicate proposal whose Role Bindings refer to Candidate Terms; it is not part of AH Memory until deterministic admission succeeds.
_Avoid_: Fact, extracted sentence

**Candidate Group**:
The set of Candidate Facts supported by one Source Span, reviewed together but admitted independently.
_Avoid_: Combined fact, paragraph

**Action Fact**:
An occurred act represented by an `ACTION` Control Template with a required actor and grounded action object; tool, location, time, purpose, and manner are optional roles from the monograph classifier.
_Avoid_: OBSERVED fallback, generic event blob

**Purpose Fact**:
A stable intended function represented by a `PURPOSE` Control Template whose `PURPOSE` role states what the subject is for.
_Avoid_: RUN, tool use, occurred action

**FOLLOW Direction**:
The temporal direction from an earlier event in `SUBJECT` to a later event in `OBJECT`.
_Avoid_: Later-to-earlier, UI-only ordering

**Memory Section Assignment**:
The classification of an admitted Candidate Fact as Common Knowledge, Private Knowledge, or Personal History; First-order Symbols are grounded in `S` independently and are never a destination of this classification.
_Avoid_: Privacy level, symbol type, document split

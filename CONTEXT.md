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

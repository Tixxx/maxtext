# Order invariants: what a proposed order must satisfy so XLA never repairs it

Every rule is stated so it can be checked from the module's graph alone. Tags:

- **ERROR**: XLA rejects the schedule or the module is invalid.
- **REPAIRED**: XLA silently rewrites the order after you hand it over. The repair
  is not what you intended and can be large.
- **HAZARD**: valid on paper, wrong results or a hang at run time.
- **PERF**: legal, but overlap or memory is lost.

`check-plan` enforces groups A and B and the pairing rules of group C (7, 8, and the
synchronous-conversion warning in 11). The rest (memory, buffers and in-place writes,
cross-rank order, side effects, annotations) is your responsibility: the tool
cannot see it.

## A. Membership (ERROR)

1. Each computation's order contains every one of its instructions exactly once:
   parameters, constants, tuples, get-tuple-element and the root included.
   Nothing missing, nothing duplicated, nothing from another computation.
2. Only non-fusion computations are ordered. Fusion bodies have no order.
3. Orders are by instruction identity. Never propose an instruction that does
   not exist in the module; XLA cannot check this safely.

## B. Dependencies (REPAIRED, then ERROR if unresolvable)

4. Every instruction comes after **all** of its operands.
5. Every instruction comes after **all** of its control predecessors. Control
   edges are not visible in an operand list: read them from the instruction text
   (`control-predecessors={...}`). Treat them exactly like operands.
6. The graph (data plus control edges) of a computation stays acyclic. You cannot
   create a cycle by reordering, but any hand-added edge could.

**What XLA does when rules 4 or 5 are broken.** It replays your order. When it
reaches an instruction whose operand or control predecessor is not placed yet, it
postpones that instruction and emits it right after its last missing predecessor.
The *user* moves later; the operand never moves earlier. Everything that depends
on the postponed instruction is postponed with it and released in a batch right
after the late predecessor, so one early misplacement can drag a whole downstream
cluster out of the order you proposed and change its internal order. An order
with no violations is reproduced exactly. Consequence: a window you scored as
hidden can silently become exposed, and compute you wanted early can end up late.
Never rely on this repair; never leave a violation in.

## C. Async pairs (ERROR, PERF)

7. **ERROR** A start has exactly one real user: its done (or a plumbing op that
   leads to it: tuple, get-tuple-element, opt-barrier, copy, parameter, while). Do
   not add a second consumer of a start's result.
8. **ERROR** A done's operand is its start (through the same plumbing). Each start
   has exactly one done. In a generic async chain (start, updates, done), keep the
   chain intact; updates are not shared between chains.
9. **ERROR** A start whose handle is carried across a while loop must stay before
   the loop-body root, and its done must stay after the instruction that reads
   the carried value. Loop structure fixes those; do not move them across it.
10. Send/recv: a send and its send-done, a recv and its recv-done keep their
    pairing and the order that the existing control edges give. Do not reorder
    pipelined send/recv chains against each other.
11. **PERF** Between a start and its done keep at least one real compute op
    (anything other than parameter, constant, bitcast, get-tuple-element). With
    none, XLA turns the pair into a synchronous collective after scheduling and
    the overlap is gone. Nested pairs with no compute between them are all
    converted. Keep a start that was already synchronous adjacent to its done.
12. **PERF** Moving a start earlier lengthens the life of its operand and output
    buffers; moving a done later lengthens its consumers' inputs. If that pushes
    peak memory over the budget XLA rematerializes or offloads, which changes the
    graph. There is no memory check in the tool.

## D. Buffers and in-place writes (HAZARD)

13. Do not move a reader of a buffer after a later in-place or aliasing write to
    the same buffer, and do not move such a write before earlier readers.
    Includes dynamic-update-slice updates, fusions and custom calls whose output
    aliases an operand, donated or aliased parameters, and the carried state of
    while loops.
14. Why this is not checked by XLA: after scheduling XLA removes copies that were
    only needed to protect such overlaps, judging them redundant from the order
    it had, and adds no control edge to keep that order. A reorder can recreate the
    conflict with no error. Preserve the original relative order of any two
    instructions that touch the same buffer or alias, and keep an in-place writer
    after every other reader of its input.
15. Host-offload (memory space S(5)) copies: a device-to-host copy start updates a
    host buffer in place; do not hoist it past other users of that buffer.
    Host-to-device copies only read it. Moving these changes memory, not results,
    as long as data edges are respected.

## E. Collectives across ranks (HAZARD)

16. Every rank must run the same order. The order must be a deterministic
    function of the module (and profile): compute it once and ship identical bytes
    to all ranks. An order produced separately per rank can deadlock.
17. With one communication stream, async collectives run in start order. Any
    single total order that is identical on all ranks is deadlock-free; reordering
    collectives is allowed if the whole module changes consistently.
18. Concurrency is where deadlock comes from. Only with multi-streaming (or
    several collective domains) do two collectives run at once; then do not
    overlap collectives whose device groups share more than one device, and treat
    unknown or empty groups as unsafe. Without multi-streaming there is no such
    constraint.
19. Do not move a collective across a barrier-like op (opt-barrier, a synchronous
    collective, a while or conditional that contains collectives) or reorder the
    pieces of a send/recv chain.

## F. Side effects (HAZARD)

20. Keep the relative order of side-effecting instructions that have no explicit
    data or control edge between them: infeed, outfeed, rng and rng state updates,
    send/recv, custom calls marked as side-effecting, collectives with a layout
    constraint or channel id. Nothing orders them for you, and the order is the
    launch order on the device.

## G. Annotations and conventions (PERF)

21. Scheduling-group annotations (`_scheduling_group_id`) are not re-checked after
    the scheduler ran. Treat a group (its starts, the compute they overlap, its
    dones) as a unit: keep the same relative layout.
22. Parameters first and the root last are not required by XLA, but leave them
    there: it is the safe choice and removes a class of surprises.
23. Peak memory and rematerialization see only the final order. A reorder that
    looks neutral can move the peak.

## Quick self-check before submitting an order

1. Same set, once each, per computation.
2. Every operand and control predecessor earlier than its user (run
   `check-plan`; do not eyeball).
3. Start before done, one done per start, at least one real op between them.
4. Nothing reading a buffer moved after a write to it; side-effecting ops in their
   original relative order.
5. Deterministic: the same inputs give the same order on every rank.

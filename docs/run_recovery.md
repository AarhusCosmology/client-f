# Continuing and recovering a run

A run can be stopped at any point and continued with the same command. What makes that safe: phase records with
the hashes of what each phase wrote, a run lock, an identity of the objective, and journals of all true evaluations.
This page describes them. The rules for changing this code are in `AGENTS.md`.

## Identity and recovery

New runs save `objective_identity.json` before any objective evaluation. Continuation
compares ordered parameters, scales, effective bounds and prior descriptors; the exact
backend input; the source files of the Cobaya components and of a likelihood function the
input names (`external:`); imported local objective modules; the loaded CLASS/CAMB/clik
modules and CAMB's shared library; and the wrapper's declared convention. Labels are not part
of scientific identity. Changes to execution or surrogate-training settings are allowed when
this identity stays unchanged. A changed native target requires a separate run.

The text of CLiENT's wrapper files (and with them Python's `abc.py`), the launcher script and
the modules beside it are not recorded (`run/identity.py`, schema 2): hashed, a comment edit in
`likelihood/cobaya.py` or in a cluster launcher's helper would refuse every run started before
it. A wrapper declares what it returns as a number, `BaseLikelihood.objective_convention`,
to be raised when a change alters the value returned at a point; a launcher's box is in the
domain and a refused value has its own record. Local dependencies reachable from the source
of an objective are an exception: their loaded modules are recorded even beside the launcher.
This includes ordinary imports, imported constants, and transitive relative imports. An
unrelated launcher helper remains excluded. A mismatch is listed entry by entry in the
error. When it is a reviewed update that leaves the objective as it was (a new CAMB build of
the same version, a moved helper), `--accept-identity-change` continues once: the earlier
record is kept in `objective_identity_history/` with the differences, the new one says what
was accepted and when, and the next start needs no flag. It does not reach the domain: a
change of bounds against the saved `metadata.json` is refused whatever is accepted. A record
of an earlier schema requires explicit review with `--accept-identity-change` before migration.
Entries removed or added by a new schema are shown rather than discarded or assumed equal:
the old record may identify an objective source under a different key. A dependency recorded
for the first time also shows as a difference and requires the same review. If the
objective actually changed, start a separate run; the flag is for verified compatible updates.
Identity history filenames are unique even for multiple accepted updates in one second.

Large external data trees and arbitrary transitive dependencies are **not** automatically
snapshotted. Put additional critical files in `likelihood.identity_files`, preferably as
absolute paths; their contents are hashed on every start. Keep the backend data installation
immutable. The manifest is an identity check of the recorded inputs, not a hermetic archive
of an entire software/data environment.

A run without an objective identity cannot be checked, so it cannot be continued: start a new
run. Multiple top-level run YAMLs are
rejected rather than chosen arbitrarily.

The master holds an exclusive OS run lock. Timestamp/name collisions refuse new-run
creation. Training CSVs, model files, histories, metrics JSON and audit artifacts use
same-directory temporary files followed by atomic replacement. `phases/` commits checksums
only after an entire initial-design, training-plus-audit, or acquisition-plus-injection
phase succeeds. Retraining archives the preceding phase record.

On continuation, a changed or missing committed artifact causes a precise refusal **before**
old training values are imported. A phase that did not reach its commit is never taken as
done: in particular, a saved model does not make an unfinished audit look complete.

A refusal here would make a job at its time limit, a `scancel` or one failed evaluation the
end of the run. Instead the next start, holding the run lock, sets the
phase aside and does it again (`PhaseStore.recover`): the files the phase may have written
are moved to `RUN/interrupted/<phase>-<time>/` under the paths they had, the marker goes to
`phases/history/` with the list, and the run starts from what the phases before it committed.
An interrupted training (with its audit) means the network is trained and audited again; an
interrupted acquisition means the saved network is sampled and a new batch evaluated. Nothing
is replayed and nothing is deleted: the true evaluations the phase had made stay in
`objective_calls/`, unused. The log says which phase was done again. An unfinished initial
design is refused, since there is no training set to continue from. A continued run
also reads the region of its saved network from `boost/region_it_N.npz` instead of building it
again; built from a training set that by then holds that network's audit points it could
differ, and the file is a committed one. Sampling can be repeated after a completed training
phase; it spends no native-objective calls. These guarantees assume normal local filesystem
lock and atomic-replacement semantics, not every distributed filesystem or hardware failure.

Augmentation's first-training marker is also written atomically. A malformed marker cannot
prove that training finished: recovery preserves it with the interrupted phase, and the
longer first-training schedule remains pending.

The branching tool, `hpc/analysis/branch_run.py`, validates the retained phase records and their artifact hashes
before creating a branch. It refuses an incomplete, malformed, missing, or damaged retained
checkpoint instead of copying its network without the record of the unfinished audit. A
later phase outside the branch can still be running. A run without phase records takes a
path that cannot certify that its phases completed.

## Call accounting and parallel failures

Ordinary pipeline evaluations are recorded under `objective_calls/rank*.jsonl`, with a
start record before invocation and either a returned value/status or an exception record.
Design, injection, acquisition, boost and audit stages are distinct. Records preserve all
finite native and derived returns, and identify nonfinite and refused results.
`objective_calls/summary.json` sums these records by stage; an unfinished call remains
visible. Its cache/import counters refer to CLiENT's ordinary pipeline, which introduces no
objective cache. Backend-internal reuse is not measured. Calls made before a run had journals
are not estimated. These are **objective-level attempts**, not counts of internal C++ computations.
Profile workflows retain their own import/cache/attempt ledgers.

Acquisition metrics count the batch before nonfinite filtering, while the final dataset
size includes subsequent injections. Use the stage journals for total objective costs,
rather than treating `n_added` or acquisition-only metrics as total work.

Ordinary MPI evaluation catches Python exceptions into structured responses, completes the
transport, and raises the batch error on every rank for the static and chunked
schedules. Inconsistent output shapes also fail collectively. The CLI aborts MPI peers on
an unhandled error elsewhere, including master-only training/setup. A segmentation fault,
hung native solver or lost host is outside Python exception recovery and still requires
MPI/scheduler supervision; no hard-crash recovery claim is made.

## Profile recovery and sampling checks

An interrupted Newton node counts earlier actual attempts against its lifetime
`maxcalls`. Exact finite cached values remain reusable without new calls, and an earlier
best native incumbent is retained when the remaining budget cannot verify another step.
Diagnostics separate previous attempts, current calls and lifetime attempts. A node that was
not interrupted is not affected. Missing interior probe values return
`unverified_interior`, retaining a finite evaluated incumbent with a failure flag rather
than attempting a model update with undefined curvature.

A torn final JSONL record is preserved verbatim with a `.truncated.json` sidecar; intact
records remain usable. One uncertain attempt is conservatively reserved against the
workflow budget. Interior corruption or a malformed newline-terminated record still
fails explicitly. Uncertain reservations are separate from confirmed backend attempts.

Configuration parsing rejects unknown keys, quoted Boolean strings, nonfinite
numbers and invalid sizes, temperatures or fractions before work begins. Remove a key it
names instead of relying on its being ignored. Use YAML `true`/`false`, not quoted strings. Scientific-notation numeric strings remain
supported because YAML loaders may parse values such as `1e-5` as strings.

AIES requires an even number of walkers, at least twice the dimension, a finite initial
ensemble of full affine rank, and usable initial target values. A collinear ensemble
cannot explore the missing directions and is refused explicitly.
